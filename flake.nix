{
  description = "ortex";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [
          rust-overlay.overlays.default
        ];

        pkgs = import nixpkgs {
          inherit overlays system;
        };

        rustVersion = pkgs.rust-bin.stable."1.77.0".default;

        common = [
          pkgs.beam.packages.erlang_26.elixir_1_16
          rustVersion
        ];

        dev =
          if builtins.getEnv "CI" != "true" then [
            pkgs.nixpkgs-fmt
            pkgs.fswatch
            pkgs.rust-analyzer
            pkgs.awscli2
            pkgs.kubernetes-helm
            pkgs.kubernetes-helmPlugins.helm-s3
          ] else [ ];

        all = common ++ dev;

        inherit (pkgs) inotify-tools terminal-notifier;
        inherit (pkgs.lib) optionals;
        inherit (pkgs.stdenv) isDarwin isLinux;

        linuxDeps = optionals isLinux [ inotify-tools ];
        darwinDeps = optionals isDarwin [ terminal-notifier ]
          ++ (with pkgs.darwin.apple_sdk.frameworks; optionals isDarwin [
          CoreFoundation
          Foundation
          CoreServices
        ]);

      in
      {
        devShells = {
          default = pkgs.mkShell {
            packages = with pkgs; all ++ linuxDeps ++ darwinDeps;
            shellHook = ''
              # this allows mix to work on the local directory
              mkdir -p .nix-mix .nix-hex
              export MIX_HOME=$PWD/.nix-mix
              export HEX_HOME=$PWD/.nix-hex
              # make hex from Nixpkgs available
              # `mix local.hex` will install hex into MIX_HOME and should take precedence
              export MIX_PATH="${pkgs.beam.packages.erlang_26.hex}/lib/erlang/lib/hex/ebin"

              export CARGO_INSTALL_ROOT=$PWD/.nix-cargo
              export CARGO_HOME=$PWD/.nix-cargo
              mkdir -p $CARGO_HOME

              export PATH=${pkgs.erlang_26}/bin:$MIX_HOME/bin:$HEX_HOME/bin:$MIX_HOME/escripts:$CARGO_HOME/bin::bin:$PATH

              export LANG=C.UTF-8
              # keep your shell history in iex
              export ERL_AFLAGS="-kernel shell_history enabled"

              export MIX_ENV=dev
            '';
          };
        };
      });
}
