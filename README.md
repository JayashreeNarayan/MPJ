# Project Instructions

## Running the MPI Version
To execute the MPI version of the code, follow these steps:

1. Run the build script:
   ```sh
   ./build.sh
   ```
2. Ensure that the YAML configuration file is correctly set up.
3. Verify that the required dependencies (e.g., Maze Poisson) are downloaded.
4. Submit the batch file for execution.

## Running the Non-MPI Version
To execute the non-MPI version, follow these steps:

1. Compile the code (only required once unless modifying the C functions):
   ```sh
   ./compile.sh
   ```
2. Ensure that the correct YAML configuration file is used (without flags for FFT and LCG).
3. Submit the batch file for execution.

## Additional Notes
- Make sure the necessary dependencies are installed before running the code.
- If you encounter issues, check the YAML configuration and ensure all required files are available.

For any further questions or troubleshooting, refer to the documentation or contact the maintainers.
