# Third-Party Notices

## Project Unlock Music Crypto

- Package: `@clamber_l/crypto`
- Version: `0.1.12`
- Copyright: Copyright (c) 2024 Project Unlock Music
- License: MIT OR Apache-2.0
- Purpose: Local Rust/WebAssembly primitives for supported protected-media
  containers.

MIT License

Copyright (c) 2024 Project Unlock Music

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The alternative Apache License 2.0 terms are included in the repository's
`LICENSE` file.

## Property-list parsing

The API uses the following MIT-licensed packages to parse the nested macOS
property lists used by the QQ Music desktop client:

- `plist 3.1.0`, Copyright (c) 2010-2017 Nathan Rajlich
- `@xmldom/xmldom 0.8.15`, Copyright 2012-present its contributors
- `base64-js 1.5.1`, Copyright (c) 2014 Jameson Little
- `xmlbuilder 15.1.1`, Copyright (c) 2013 Ozgur Ozcitak

Each package is distributed under the MIT License reproduced above.

## MuScriptor

- Package: `muscriptor`
- Version: `0.3.0`
- Authors: Kyutai and Mirelo
- Code license: MIT
- Purpose: Optional local high-accuracy multi-instrument transcription
- Source: <https://github.com/muscriptor/muscriptor>

The optional Python runtime is distributed under the MIT License reproduced
above. MuScriptor model weights are separate gated downloads published under
CC BY-NC 4.0. They are not included in this repository or application
installer. The desktop requires the user to acknowledge that noncommercial
license before enabling or importing those weights.

License text and terms for the weights:
<https://creativecommons.org/licenses/by-nc/4.0/>

## Noise2Fret

- Source: `RiccardoVib/Noise2Fret`
- Authors: Riccardo Simionato and Louis Bigo
- Code license: MIT
- Model-weight license: CC BY 4.0
- Purpose: Optional playability-aware guitar string/fret refinement

The Worker contains an inference-only adaptation of the published network
layout. Training, dataset and plotting code are not included. The model
checkpoint is downloaded separately from Zenodo, verified before activation
and stored in private application data.

Source: <https://github.com/RiccardoVib/Noise2Fret>

Weights: <https://doi.org/10.5281/zenodo.23037260>

## MR-MT3 and MT3-Infer

- Package: `mt3-infer`
- Version: `0.2.0`
- Package license: MIT
- Active adapter: MR-MT3
- MR-MT3 code and checkpoint license: MIT
- Purpose: Optional secondary event verification in Medium/Large modes

Only the MR-MT3 adapter is enabled and collected into the open-source desktop
Worker. The MT3-PyTorch and YourMT3 adapters present in the upstream package
are explicitly excluded from the bundle because their upstream licensing is
not suitable for this distribution policy. The MR-MT3 checkpoint is downloaded
separately and SHA-256 verified before activation.

Adapter source: <https://github.com/openmirlab/mt3-infer>

Model source: <https://github.com/gudgud96/MR-MT3>
