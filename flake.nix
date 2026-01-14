{
  description = "bp-k-means development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python313
            uv
            gnumake
            gh
            jq
            git
            curl
            # Libraries often needed by Python packages
            stdenv.cc.cc.lib
            zlib
          ];

          # Set LD_LIBRARY_PATH to help python wheels find system libraries
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];

          shellHook = ''
            echo "Environment loaded"
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"
          '';
        };
      });
}
