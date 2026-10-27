# Third-party notices

BallaSim is licensed under CC BY-NC 4.0 (legal code in `LICENSE`; copyright, patents and attribution in `NOTICE`). Apart
from the ported routines listed under [Ported code](#ported-code) (C++ translations of NumPy, SciPy, LAPACK, OpenBLAS
and CPython code, and the AEDAT4 writer's independent implementation of FlatBuffers' and dv-processing's formats,
which stay under their own licences, reproduced in this file), no third-party source code is in this repository. The
other components below are either linked from the conda environment, used only at build or test time, or contribute
published numerical constants.

The IMU model (`src/imu`) and the run hook use published sensor constants and the documented model in
`docs/design/phase3/outputs.md`; the hook adds no library dependency or third-party source. IMU HDF5 output uses the
same h5py/HDF5 dependency as event output, with Fletcher-32 and no compression plugin.

## Linked or embedded in the extension modules

| Component | Use | Licence |
| --- | --- | --- |
| [Embree](https://github.com/RenderKit/embree) 4.4 | CPU ray tracing (shared library from conda-forge) | Apache-2.0 |
| [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) 0.3.34 with reference LAPACK | BLAS and LAPACK of `_scene` (shared library from conda-forge) | BSD-3-Clause (LAPACK: modified BSD, below) |
| [libpng](http://www.libpng.org/pub/png/libpng.html) 1.6 (with zlib) | PNG decoding of `_scene`'s PLY import (shared library from conda-forge) | PNG Reference Library License version 2 (zlib: zlib License) |
| [nanobind](https://github.com/wjakob/nanobind) | Python bindings (compiled statically into each module) | BSD-3-Clause |
| [oneTBB](https://github.com/uxlfoundation/oneTBB) 2023 | Embree's task scheduler, linked with it by `_core`, `_render` and `_engine` (shared library from conda-forge; PHASE3.md D-P3-7) | Apache-2.0 (text [below](#apache-license-version-20)) |
| [LZ4](https://github.com/lz4/lz4) 1.10 | LZ4 frames of `_export` (shared library from conda-forge) | BSD-2-Clause |
| [Zstandard](https://github.com/facebook/zstd) 1.5 | Zstandard frames of `_export` (shared library from conda-forge) | BSD-3-Clause |
| [bzip2](https://sourceware.org/bzip2/) 1.0.8 | bzip2 streams of `_export` (shared library from conda-forge) | bzip2-1.0.6 licence (BSD-style) |
| CUDA runtime (`cudart_static`) | Linked into `_cuda` and `_optix` | NVIDIA CUDA Toolkit EULA (redistributable runtime component) |
| NVIDIA OptiX 9.1 SDK headers | Compiled into `_optix` only when the user points `OPTIX_ROOT` at an SDK whose licence they accepted. Never vendored, copied or downloaded; the runtime ships with the NVIDIA driver | NVIDIA OptiX SDK licence |

## Ported code

The scene side (`src/scene`, docs/dev/CPP_PORT.md item 5) reproduces the Python EVIMO2 trajectory fitter bit for bit,
so some routines are C++ translations of the libraries the Python calls, following their loops and order of operations.
The AEDAT4 writer (`src/export`, P3-A A8) is an independent implementation of FlatBuffers' buffer layout and
dv-processing's AEDAT4 format from their Apache-2.0 schemas and sources, not a clean-room one (PHASE3.md D-P3-7), so
its three files are listed too. Each file below says so in its first lines. These translations are not covered by
CC BY-NC 4.0: each stays under the licence of the code it is derived from, whose notice follows.

| File | Derived from |
| --- | --- |
| `src/scene/npo.cpp` | NumPy: `pairwise_sum_DOUBLE` (the pairwise sum of `np.sum`), and the operation order of `np.median`, `np.percentile` (linear), `np.std` and `np.arange` |
| `src/scene/sparse.cpp` | SciPy `scipy/sparse/sparsetools`: `csr_tocsc`, `csr_matmat` (SMMP), `coo_tocsr`, `csr_binop_csr` and `csr_sum_duplicates` |
| `src/scene/rotation.cpp` | SciPy 1.18 `scipy/spatial/transform/_rotation_cy.pyx` (the Cython backend of `Rotation`: `from_matrix`, `as_matrix`, `as_rotvec`, `from_rotvec`, composition, `mean`, with SciPy's error messages) and `_rotation.py` (`Slerp`); NumPy's einsum sum-of-products kernels (their summation order) |
| `src/scene/traj/banded.cpp` | Reference LAPACK `dpbtf2` (band Cholesky); OpenBLAS's `dsyr`/`daxpy` kernel split; NumPy's einsum sum-of-products kernels and buffering (their summation order) |
| `src/scene/blas.cpp` | OpenBLAS's Haswell `dgemm` kernel (its accumulation order for 3x3 products) |
| `src/scene/evimo2/build.cpp` | NumPy's einsum sum-of-products kernel `contig_contig_outstride0_two` (its two-lane summation order) |
| `src/scene/npla.cpp` | NumPy: `matmul`'s BLAS dispatch (`matmul.c.src`: `is_blasable2d`, the gemm / gemv / dot choice and their arguments), `umath_linalg`'s `det` (`slogdet` from the LU factors) and LAPACK layouts, `np.linspace` |
| `src/scene/geometry.cpp` | NumPy: the operation order of `np.cross`, `np.unique(axis=0)` and `np.bincount` as the adapter and the mesh helpers use them |
| `src/scene/kdtree.cpp` | SciPy `scipy/spatial/ckdtree`: the summation order of `sqeuclidean_distance_double` |
| `src/scene/pyfmt.cpp` | CPython: `format_float_short` (`Python/pystrtod.c`, `repr` of a float) and `float.hex` (`Objects/floatobject.c`), for doubles only; the changes: rewritten in C++ over `std::to_chars` digits, other modes and types left out |
| `src/export/flatbuf_lite.h` | FlatBuffers (google/flatbuffers): the buffer layout its `FlatBufferBuilder` produces (back-to-front building, alignment from the end, shared vtables, forced defaults, `Finish` with the identifier and size prefix), as the R-FMT prototype emulated version 1.10; written independently, with no FlatBuffers code (the AEDAT4 writer, outputs.md A8) |
| `src/export/aedat4.{h,cpp}` | dv-processing (iniVation): its AEDAT4 FlatBuffers schemas (`IOHeader`, `FileDataTable`, `Event`, `IMU`: field ids, types, defaults and the field order of the generated `Create*` functions) and its writer's behaviour (the size-prefixed FileDataTable compressed with the file's codec, the `dataTablePosition` patched at close, the codec settings, the IMU units); written independently, with no dv-processing code |

**NumPy** (BSD-3-Clause):

> Copyright (c) 2005-2025, NumPy Developers. All rights reserved.
>
> The einsum sum-of-products kernels (`numpy/_core/src/multiarray/einsum_sumprod.c.src`): Copyright (c) 2011 by Mark
> Wiebe (mwwiebe@gmail.com), The University of British Columbia; under the NumPy licence below.
>
> Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
> following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
>   disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
>   disclaimer in the documentation and/or other materials provided with the distribution.
> * Neither the name of the NumPy Developers nor the names of any contributors may be used to endorse or promote
>   products derived from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
> INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
> WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**SciPy** (BSD-3-Clause):

> Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers. All rights reserved.
>
> Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
> following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
>    disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
>    following disclaimer in the documentation and/or other materials provided with the distribution.
> 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
>    products derived from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
> INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
> WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**LAPACK** (modified BSD):

> Copyright (c) 1992-2023 The University of Tennessee and The University of Tennessee Research Foundation. All rights
> reserved.
> Copyright (c) 2000-2023 The University of California Berkeley. All rights reserved.
> Copyright (c) 2006-2023 The University of Colorado Denver. All rights reserved.
>
> Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
> following conditions are met:
>
> - Redistributions of source code must retain the above copyright notice, this list of conditions and the following
>   disclaimer.
> - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
>   disclaimer listed in this license in the documentation and/or other materials provided with the distribution.
> - Neither the name of the copyright holders nor the names of its contributors may be used to endorse or promote
>   products derived from this software without specific prior written permission.
>
> The copyright holders provide no reassurances that the source code provided does not infringe any patent, copyright,
> or any other intellectual property rights of third parties. The copyright holders disclaim any liability to any
> recipient for claims brought against recipient by any third party for infringement of that parties intellectual
> property rights.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
> INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
> WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**CPython** (Python Software Foundation License Version 2):

> Copyright (c) 2001-2023 Python Software Foundation; All Rights Reserved.
>
> 1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"), and the Individual or Organization
>    ("Licensee") accessing and otherwise using this software ("Python") in source or binary form and its associated
>    documentation.
> 2. Subject to the terms and conditions of this License Agreement, PSF hereby grants Licensee a nonexclusive,
>    royalty-free, world-wide license to reproduce, analyze, test, perform and/or display publicly, prepare derivative
>    works, distribute, and otherwise use Python alone or in any derivative version, provided, however, that PSF's
>    License Agreement and PSF's notice of copyright, i.e., "Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007,
>    2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023 Python Software
>    Foundation; All Rights Reserved" are retained in Python alone or in any derivative version prepared by Licensee.
> 3. In the event Licensee prepares a derivative work that is based on or incorporates Python or any part thereof, and
>    wants to make the derivative work available to others as provided herein, then Licensee hereby agrees to include
>    in any such work a brief summary of the changes made to Python.
> 4. PSF is making Python available to Licensee on an "AS IS" basis. PSF MAKES NO REPRESENTATIONS OR WARRANTIES,
>    EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR
>    WARRANTY OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON WILL NOT INFRINGE ANY
>    THIRD PARTY RIGHTS.
> 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL
>    DAMAGES OR LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON, OR ANY DERIVATIVE THEREOF, EVEN
>    IF ADVISED OF THE POSSIBILITY THEREOF.
> 6. This License Agreement will automatically terminate upon a material breach of its terms and conditions.
> 7. Nothing in this License Agreement shall be deemed to create any relationship of agency, partnership, or joint
>    venture between PSF and Licensee. This License Agreement does not grant permission to use PSF trademarks or trade
>    name in a trademark sense to endorse or promote products or services of Licensee, or any third party.
> 8. By copying, installing or otherwise using Python, Licensee agrees to be bound by the terms and conditions of this
>    License Agreement.

**FlatBuffers** (Google) and **dv-processing** (iniVation), Apache License 2.0, whose full text is reproduced
verbatim under [Apache License, Version 2.0](#apache-license-version-20) at the end of this file
(`src/export/flatbuf_lite.h` and `src/export/aedat4.{h,cpp}` say in their first lines what they derive from):

> Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
> the License. You may obtain a copy of the License at
>
>     http://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
> an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
> specific language governing permissions and limitations under the License.

**OpenBLAS** (BSD-3-Clause):

> Copyright (c) 2011-2014, The OpenBLAS Project. All rights reserved.
>
> Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
> following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
>    disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
>    following disclaimer in the documentation and/or other materials provided with the distribution.
> 3. Neither the name of the OpenBLAS project nor the names of its contributors may be used to endorse or promote
>    products derived from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
> INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
> WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Build and test tools

scikit-build-core (Apache-2.0), CMake (BSD-3-Clause), Ninja (Apache-2.0), doctest (MIT),
pytest (MIT), hypothesis (MPL-2.0), NumPy and SciPy (BSD-3-Clause).

## Numerical constants and test data

- **Sobol direction numbers** (`src/core/rng/sobol.h`): the primitive polynomials and initial
  direction numbers of dimensions 2-8 of `new-joe-kuo-6.21201`, from
  <https://web.maths.unsw.edu.au/~fkuo/sobol/>. Copyright (c) 2008, Frances Y. Kuo and Stephen Joe.
  All rights reserved. Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met: (1) redistributions
  of source code must retain the above copyright notice, this list of conditions and the following
  disclaimer; (2) redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or other materials
  provided with the distribution; (3) neither the names of the copyright holders nor the names of
  the University of New South Wales and the University of Waikato and its contributors may be used
  to endorse or promote products derived from this software without specific prior permission.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES
  ARE DISCLAIMED.
- **Philox4x32-10** (`src/core/rng/philox.h`): algorithm and constants from Salmon, Moraes, Dror
  and Shaw, "Parallel random numbers: as easy as 1, 2, 3", SC'11. The known-answer vectors in the
  tests are from Random123 (BSD-3-Clause, D. E. Shaw Research).
- **Owen scrambling**: hash-based construction of Burley, "Practical Hash-based Owen Scrambling",
  JCGT 9(4), 2020, with the Laine-Karras permutation constants of N. Vegdahl, "Building a Better
  LK Hash" (2021), and the `lowbias32` integer hash of C. Wellons (public domain).
- **Cycles timing** (`src/frames/cycles_timing.{h,cpp}`): the flat shutter table's bounds and
  centre offset (`intern/cycles/scene/camera.cpp`), the rolling-shutter ray time
  (`kernel/camera/camera.h`), the motion-step node times and counts (`blender/object.cpp`,
  `blender/util.h`, `scene/object.cpp`) and the rotation interpolation rule (`util/transform.h`)
  of Blender's Cycles renderer v4.2.1, re-implemented so that frames rendered by Cycles can be
  timed on the event clock. Cycles is Copyright 2011-2022 Blender Foundation (the SPDX headers of
  the cited v4.2.1 files), licensed under the Apache License, Version 2.0, reproduced under
  [Apache License, Version 2.0](#apache-license-version-20) below.

## Documentation figures

- No EVIMO2 data, statistics plots, event images or mesh renders are in this repository; plots
  comparing against EVIMO2 stay with the local run outputs, except the README figures: the teaser video (`docs/media/teaser_sfm.webp`) and `docs/images/events_xyt_scene13.jpg` show EVIMO2 real events beside BallaSim's
  simulation, with the EVIMO2 authors' permission.
- `docs/images/p1_lab_frame.png` is rendered from the CC0 (Poly Haven, ambientCG) and CC BY 4.0 (YCB
  Object and Model Set, Calli et al.; Google Scanned Objects, Google Research) assets listed, with
  the changes made to them, in the header of `scenes/lab_tabletop.yaml`.

## Apache License, Version 2.0

The licence of the files derived from FlatBuffers and dv-processing (see [Ported code](#ported-code)), verbatim
(https://www.apache.org/licenses/LICENSE-2.0.txt):

```text

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
