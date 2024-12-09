{
  description = "University: Natural Language Processing: Project (2024/11/20--2024/12/18)";

  inputs = {
    asciidoctor-nix.url = "github:trueNAHO/asciidoctor.nix";
    flake-utils.follows = "asciidoctor-nix/flake-utils";
    nixpkgs.follows = "asciidoctor-nix/nixpkgs";
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystemPassThrough (
      system: let
        lib = inputs.asciidoctor-nix.mkLib pkgs.lib;
        pkgs = inputs.nixpkgs.legacyPackages.${system};
      in
        lib.asciidoctor.mergeAttrsMkMerge [
          (
            inputs.flake-utils.lib.eachDefaultSystem (
              _: {
                packages = lib.fix (
                  self: {
                    application-default = pkgs.stdenvNoCC.mkDerivation {
                      buildPhase = let
                        out = "${builtins.placeholder "out"}/share/src";
                      in ''
                        mkdir --parents ${out}
                        zip --recurse-paths ${out}/application.zip .
                      '';

                      name = "application-default";
                      nativeBuildInputs = [pkgs.zip];
                      postPatch = "cp ${./LICENSE} LICENSE";
                      src = src/application;
                    };

                    application-default-external = self.application-default;

                    default = pkgs.buildEnv {
                      name = "default";

                      paths = lib.attrsets.attrValues (
                        lib.filterAttrs
                        (
                          package: _:
                            builtins.match ".*-default" package != null
                        )
                        inputs.self.packages.${system}
                      );
                    };

                    default-external = pkgs.buildEnv {
                      name = "default-external";

                      paths = lib.attrsets.attrValues (
                        lib.filterAttrs
                        (
                          package: _:
                            builtins.match ".*-default-external" package != null
                        )
                        inputs.self.packages.${system}
                      );
                    };
                  }
                );
              }
            )
          )

          (
            inputs.asciidoctor-nix.mkOutputs {
              checks.hooks = let
                excludes.marimo = [''^src/application/src/main\.py$''];
              in {
                autoflake.enable = true;
                isort.enable = true;

                mypy = {
                  enable = true;
                  excludes = excludes.marimo;
                };

                pyright = {
                  enable = true;
                  excludes = excludes.marimo;
                };

                ruff-format.enable = true;
                ruff.enable = true;
              };

              devShells.packages = with pkgs; [
                (python3.withPackages (_: []))
                marimo
              ];

              packages = {
                inherit (inputs.self) lastModified;

                inputFile = "pages/index.adoc";
                name = "presentation";
                src = src/presentation;
              };
            }
          )
        ]
    );
}
