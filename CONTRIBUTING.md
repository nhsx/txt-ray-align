# Contribution guidelines
Thank you for your interest in contributing to this project! We've compiled these docs to help you understand our contribution guidelines. If you still have questions, please [contact us](mailto:analytics-unit@nhsx.nhs.uk), we'd be happy to help.

## Code of Conduct
Please read our [code of conduct](./CODE_OF_CONDUCT.md) before contributing.

## Contributing
If you’ve got an idea, suggestion, or encountered a bug, you can create a GitHub issue.

When raising bugs please explain the issue in good detail and provide a guide to how to replicate it.

Please raise feature requests as issues before contributing any code. This ensures they are discussed properly before any time is spent on them.

## Updating the Changelog

If you open a GitHub pull request on this repo, please update `CHANGELOG` to reflect your contribution.

Add your entry under `Unreleased` as: 
- `Breaking changes`
- `New features`
- `Fixes`

Internal changes to the project that are not part of the public API do not need changelog entries, for example fixing the CI build server.

These sections follow [semantic versioning](https://semver.org/spec/v2.0.0.html), where:

- `Breaking changes` corresponds to a `major` (1.X.X) change.
- `New features` corresponds to a `minor` (X.1.X) change.
- `Fixes` corresponds to a `patch` (X.X.1) change.

See the [changelog](./CHANGELOG.md) for an example of how this looks.
