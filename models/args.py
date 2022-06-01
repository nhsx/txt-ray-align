from argparse import ArgumentParser
from pytorch_lightning import Trainer


def parse_arguments():

    parser = ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_img', type=float, default=None, help='override learning rate for image encoder')
    parser.add_argument('--lr_txt', type=float, default=None, help='override learning rate for text encoder')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='number of epochs for linear warmup')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Choose optimizer from AdamW or SGD')

    parser.add_argument('--use_clip', action='store_true', default=False, help='set to use CLIP-trained RN50 - this overrides remaining args in this block')
    parser.add_argument('--image_encoder', type=str, default=None, help='name of image encoder')
    parser.add_argument('--text_encoder', type=str, default=None, help='name of text encoder')
    parser.add_argument('--add_projection', action='store_true', default=False, help='add linear projection to text encoder')
    parser.add_argument('--embed_dim', type=int, default=768, help='output dimension of encoders')
    parser.add_argument('--freeze_img_encoder', action='store_true', default=False, help='set to only train output layer')
    parser.add_argument('--freeze_txt_encoder', action='store_true', default=False, help='set to only train output layer')
    parser.add_argument('--freeze_layers', type=int, default=-1, help='max index of text layer to freeze, -1 meaning all')
    parser.add_argument('--use_pretrained', action='store_true', default=False, help='set to load pretrained image model')
    parser.add_argument('--local_files_only', action='store_true', default=False, help='set to look for text model in local directory ./<dir>/<name>')
    
    parser.add_argument('--checkpoint_file', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--checkpoint_every', type=int, default=0, help='number of steps between checkpointing - in addition to per epoch checkpoints')

    parser.add_argument('--image_folder', type=str, default=None, help='directory of your image training folder')
    parser.add_argument('--text_folder', type=str, default=None, help='directory of your text training folder')
    parser.add_argument('--image_folder_val', type=str, default=None, help='directory of your image val folder')
    parser.add_argument('--text_folder_val', type=str, default=None, help='directory of your text val folder')

    parser.add_argument('--train', type=str, default=None, help='path to train csv')
    parser.add_argument('--val', type=str, default=None, help='path to val csv')

    parser.add_argument('--batch_size', type=int, default=8, help='size of the batch')
    parser.add_argument('--minibatch_size', type=int, default=0, help='microbatches')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
    parser.add_argument('--image_size', type=int, default=224, help='size of the images')
    parser.add_argument('--resize_ratio', type=float, default=0.75, help='minimum size of images during random crop')
    parser.add_argument('--num_sentences', type=int, default=1, help='choose number of sentences to sample')
    parser.add_argument('--shuffle', action='store_true', default=False, help='whether to use shuffling during sampling')

    parser.add_argument('--use_teacher', action='store_true', default=False, help='whether to use self distillation during training')

    parser = Trainer.add_argparse_args(parser)
    return parser.parse_args()
