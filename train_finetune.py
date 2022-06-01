import clip

from data.text_image_dm import MIMICDataModule
from models import CustomCLIPWrapper, init_img_model, init_txt_model, parse_arguments
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def main(hparams):

    using_clip = False
    if hparams.use_clip:
        if hparams.image_encoder is not None or hparams.text_encoder is not None:
            print("Warning - image_encoder and text_encoder args unused")

        using_clip = True
        clp, preprocess = clip.load("RN50", device="cpu")

        for p in clp.parameters():
            p.data = p.data.float()
            if p.grad:
                p.grad.data = p.grad.data.float()

        img_encoder = clp.visual
        txt_encoder = clp.transformer

    else:
        if hparams.image_encoder is None or hparams.text_encoder is None:
            print("Please --use_clip or set image and text encoders, exiting...")
            exit(1)

        img_encoder, _ = init_img_model(
            hparams.image_encoder,
            hparams.embed_dim,
            hparams.freeze_img_encoder,
            hparams.use_pretrained,
        )
        txt_encoder, tokenizer = init_txt_model(
            hparams.text_encoder,
            hparams.embed_dim,
            hparams.freeze_txt_encoder,
            hparams.freeze_layers,
            hparams.add_projection,
            hparams.local_files_only,
        )

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    if hparams.checkpoint_file is not None:
        model = CustomCLIPWrapper.load_from_checkpoint(
            checkpoint_path=hparams.checkpoint_file,
            image_encoder=img_encoder,
            text_encoder=txt_encoder,
        )
    else:
        model = CustomCLIPWrapper(
            img_encoder,
            txt_encoder,
            hparams.minibatch_size,
            using_clip=using_clip,
            lr=hparams.lr,
            lr_img=hparams.lr_img,
            lr_txt=hparams.lr_txt,
            warmup_epochs=hparams.warmup_epochs,
            use_teacher=hparams.use_teacher,
        )
        if using_clip:
            model.finish_clip_init(clp)

    dm = MIMICDataModule.from_argparse_args(
        hparams, custom_tokenizer=None if using_clip else tokenizer
    )

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor="val_loss", save_top_k=1))

    if hparams.checkpoint_every > 0:
        callbacks.append(ModelCheckpoint(every_n_train_steps=hparams.checkpoint_every))

    if int(hparams.devices) > 1:
        trainer = Trainer.from_argparse_args(
            hparams, accelerator="gpu", strategy="ddp", callbacks=callbacks
        )
    else:
        trainer = Trainer.from_argparse_args(
            hparams, accelerator="auto", callbacks=callbacks
        )

    if hparams.checkpoint_file is not None:
        trainer.fit(model, dm, ckpt_path=hparams.checkpoint_file)
    else:
        trainer.fit(model, dm)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
