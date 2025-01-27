{pyproject-nix, pyproject-build-systems, pyproject-overrides, lib, workspace, python3, callPackage}:
let overlay = workspace.mkPyprojectOverlay {
      # Prefer prebuilt binary wheels as a package source.
      # Sdists are less likely to "just work" because of the metadata missing from uv.lock.
      # Binary wheels are more likely to, but may still require overrides for library dependencies.
      sourcePreference = "wheel"; # or sourcePreference = "sdist";
      # Optionally customise PEP 508 environment
      # environ = {
      #   platform_release = "5.10.65";
      # };
    };
    fix-vectorlink-gpu-build = final: prev: {
      vectorlink-gpu = prev.vectorlink-gpu.overrideAttrs (old: {
        buildInputs = old.buildInputs or [] ++ [final.hatchling final.pathspec final.pluggy final.packaging final.trove-classifiers];
      });
      vectorlink-py = prev.vectorlink-py.overrideAttrs (old: {
        buildInputs = old.buildInputs or [] ++ [final.hatchling final.pathspec final.pluggy final.packaging final.trove-classifiers];
      });
    };
in
(callPackage pyproject-nix.build.packages {
  python = python3;
}).overrideScope (
  lib.composeManyExtensions [
    pyproject-build-systems.overlays.default
    overlay
    pyproject-overrides.cuda
    pyproject-overrides.default
    fix-vectorlink-gpu-build
  ]
)
