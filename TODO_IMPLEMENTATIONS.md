| File Path                                                     | Function/Method Name                       | Brief Description of Placeholder Status                                  |
| ------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------ |
| `reconlib/csm.py`                                             | `estimate_espirit_maps`                    | Placeholder function, returns zeros.                                     |
| `reconlib/phase_unwrapping/puror.py`                          | `unwrap_phase_puror`                       | Placeholder function, returns input.                                     |
| `reconlib/phase_unwrapping/romeo.py`                          | `unwrap_phase_romeo`                       | Placeholder function, returns input.                                     |
| `reconlib/phase_unwrapping/deep_learning_unwrap.py`           | `unwrap_phase_deep_learning`               | Placeholder function, returns input. U-Net model needs definition/loading. |
| `reconlib/b0_mapping/b0_nice.py`                              | `calculate_b0_map_nice`                    | Placeholder function, returns zeros.                                     |
| `reconlib/coil_combination.py`                                | `regrid_kspace`                            | Raises `NotImplementedError`.                                            |
| `reconlib/coil_combination.py`                                | `filter_low_frequencies`                   | Raises `NotImplementedError`.                                            |
| `reconlib/coil_combination.py`                                | `extract_calibration_region`               | Raises `NotImplementedError`.                                            |
| `reconlib/coil_combination.py`                                | `compute_espirit_kernel`                   | Raises `NotImplementedError`.                                            |
| `reconlib/coil_combination.py`                                | `apply_espirit_kernel`                     | Raises `NotImplementedError`.                                            |
| `reconlib/coil_combination.py`                                | `compute_voronoi_tessellation`             | Raises `NotImplementedError`.                                            |
| `reconlib/coil_combination.py`                                | `compute_polygon_area`                     | Raises `NotImplementedError`.                                            |
| `reconlib/io.py`                                              | `load_ismrmrd`                             | Raises `NotImplementedError`.                                            |
| `reconlib/io.py`                                              | `save_ismrmrd`                             | Raises `NotImplementedError`.                                            |
| `reconlib/io.py`                                              | `load_nifti_complex`                       | Raises `NotImplementedError`.                                            |
| `reconlib/io.py`                                              | `save_nifti_complex`                       | Raises `NotImplementedError`.                                            |
| `reconlib/io.py`                                              | `DICOMIO.read`                             | Raises `NotImplementedError`.                                            |
| `reconlib/io.py`                                              | `DICOMIO.write`                            | Raises `NotImplementedError`.                                            |
| `reconlib/optimizers.py`                                      | `PenalizedLikelihoodReconstruction._data_fidelity_gradient` | Raises `NotImplementedError`. Gradient formula needs implementation.     |
| `reconlib/optimizers.py`                                      | `PenalizedLikelihoodReconstruction.solve`  | Raises `NotImplementedError`. Needs integration with optimizer.          |
| `reconlib/optimizers.py`                                      | `convergence_monitor`                      | Raises `NotImplementedError`. Logic not fully implemented.               |
| `reconlib/optimizers.py`                                      | `metrics_calculator`                       | Raises `NotImplementedError`. Metrics calculation is placeholder.        |
| `reconlib/pet_ct_pipeline.py`                                 | `ReconstructionPipeline.run`               | Raises `NotImplementedError`. Core optimizer call needs implementation.  |
| `reconlib/pet_ct_preprocessing.py`                            | `normalize_counts`                         | Raises `NotImplementedError`.                                            |
| `reconlib/pet_ct_preprocessing.py`                            | `randoms_correction`                       | Raises `NotImplementedError`.                                            |
| `reconlib/pet_ct_preprocessing.py`                            | `normalize_projection_data`                | Raises `NotImplementedError`.                                            |
| `reconlib/pet_ct_simulation.py`                               | `PhantomGenerator.generate`                | Raises `NotImplementedError` for most phantom types.                   |
| `reconlib/physics.py`                                         | `AttenuationCorrection.apply`              | Raises `NotImplementedError`.                                            |
| `reconlib/physics.py`                                         | `ScatterCorrection.correct`                | Raises `NotImplementedError`.                                            |
| `reconlib/physics.py`                                         | `DetectorResponseModel.apply`              | Raises `NotImplementedError`.                                            |
| `reconlib/deeplearning/models/unet_denoiser.py`               | Module-level                               | Empty file (`pass`). U-Net model architecture undefined.               |
| `reconlib/deeplearning/losses/common_losses.py`             | Module-level                               | Empty file (`pass`). Common loss functions undefined.                  |
| `reconlib/metrics/__init__.py`                                | Module-level                               | Empty file (`pass`). Metrics module initialization.                    |
| `reconlib/deeplearning/layers/__init__.py`                    | Module-level                               | Empty file (`pass`). Layers module initialization.                     |
| `reconlib/deeplearning/utils/__init__.py`                     | Module-level                               | Empty file (`pass`). DL utils module initialization.                   |
