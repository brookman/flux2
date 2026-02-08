{
  description = "FLUX.2 - Frontier Visual Intelligence with CUDA support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;  # Required for CUDA
            cudaSupport = true;
          };
        };

        # Python environment - minimal, let pip install PyTorch with CUDA
        pythonEnv = pkgs.python312.withPackages (ps: [
          # Build tools
          ps.pip
          ps.setuptools
          ps.wheel
          ps.virtualenv
          ps.diffusers
        ]);

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
            pkgs.git
            pkgs.gcc
            pkgs.stdenv.cc.cc.lib
          ];

          shellHook = ''
            echo "FLUX.2 Development Environment"
            echo "================================"

            # Set up LD_LIBRARY_PATH for CUDA libraries
            # On NixOS, NVIDIA drivers are in /run/opengl-driver
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH"

            # Set CUDA environment variables
            export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
            export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"

            # Create and activate virtual environment
            if [ ! -d ".venv" ]; then
              echo "Creating virtual environment..."
              python -m venv .venv
            fi

            source .venv/bin/activate

            # Install dependencies if not already installed
            if ! python -c "import torch" 2>/dev/null; then
              echo ""
              echo "Installing Python dependencies..."
              echo "This may take a few minutes on first run..."
              pip install --upgrade pip
              pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
            fi

            # Add src to PYTHONPATH for convenience
            export PYTHONPATH="$PWD/src:$PYTHONPATH"

            echo ""
            echo "Python: $(python --version)"
            if python -c "import torch" 2>/dev/null; then
              echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
              echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
              echo "CUDA Version: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")')"
            fi
            echo ""
            echo "Environment variables for model paths:"
            echo "  KLEIN_9B_MODEL_PATH - Path to FLUX.2 [klein] 9B model"
            echo "  KLEIN_9B_BASE_MODEL_PATH - Path to FLUX.2 [klein] 9B Base model"
            echo "  AE_MODEL_PATH - Path to autoencoder model"
            echo ""
            echo "Run the CLI with:"
            echo "  ./run.sh"
            echo "  or: PYTHONPATH=src python scripts/cli.py"
            echo ""
          '';

          # Environment variables
          CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
          EXTRA_LDFLAGS = "-L${pkgs.cudaPackages.cudatoolkit}/lib";
          EXTRA_CCFLAGS = "-I${pkgs.cudaPackages.cudatoolkit}/include";
        };

        # Package definition (builds the package without heavy deps)
        packages.default = pkgs.python312Packages.buildPythonPackage {
          pname = "flux";
          version = "0.1.0";
          src = ./.;
          format = "pyproject";

          propagatedBuildInputs = with pkgs.python312Packages; [
            setuptools
            wheel
          ];

          # Don't run tests during build (they may require models)
          doCheck = false;

          meta = with pkgs.lib; {
            description = "Inference codebase for FLUX.2";
            license = licenses.unfree;
            platforms = platforms.linux;
          };
        };
      }
    );
}
