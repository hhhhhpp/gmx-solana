/// Configuration.
pub mod config;

/// Utils for wallet.
pub mod wallet;

/// Commands.
pub mod commands;

use std::{ops::Deref, path::PathBuf};

use clap::Parser;
use commands::{Command, CommandClient, Commands, Context};
use config::Config;
use eyre::OptionExt;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};

const ENV_PREFIX: &str = "GMSOL_";
const CONFIG_DIR: &str = "gmsol";

/// We use `__` in the name of environment variable as an alias of `.`.
///
/// See [`Env`] for more infomation.
const DOT_ALIAS: &str = "__";

/// Command-line interface for GMX-Solana.
#[derive(Debug)]
pub struct Cli(Inner);

impl Cli {
    /// Creates from the command line arguments.
    pub fn init() -> eyre::Result<Self> {
        let cli = Inner::parse();

        let config_path = cli.find_config()?;
        let Inner {
            config,
            command,
            verbose,
            ..
        } = cli;

        let config = Figment::new()
            .merge(Serialized::defaults(config))
            .join(Env::prefixed(ENV_PREFIX).split(DOT_ALIAS))
            .join(Toml::file(config_path.clone()))
            .extract()?;

        Ok(Self(Inner {
            config_path: Some(config_path),
            config,
            command,
            verbose,
        }))
    }
}

impl Deref for Cli {
    type Target = Inner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Command-line interface for GMX-Solana.
#[derive(Debug, Parser)]
#[command(
    version = concat!(
        env!("CARGO_PKG_VERSION"), " (",
        env!("VERGEN_BUILD_DATE"), ")"
    ),
    long_version = concat!(
        env!("CARGO_PKG_VERSION"), "\n",
        "Built: ", env!("VERGEN_BUILD_TIMESTAMP"), "\n",
        "Git commit: ", env!("VERGEN_GIT_SHA"), "\n",
        "Rustc version: ", env!("VERGEN_RUSTC_SEMVER"), "\n",
        "Enabled features: ", env!("VERGEN_CARGO_FEATURES"), "\n",
        "Debug: ", env!("VERGEN_CARGO_DEBUG"),
    ),
    about = None,
    long_about = None,
)]
pub struct Inner {
    /// Path to the config file.
    #[clap(long = "config", short)]
    config_path: Option<PathBuf>,
    /// Enable detailed output.
    #[clap(long, short, global = true)]
    verbose: bool,
    /// Config.
    #[command(flatten)]
    config: Config,
    /// Commands.
    #[command(subcommand)]
    command: Commands,
}

impl Inner {
    fn find_config(&self) -> eyre::Result<PathBuf> {
        use etcetera::{choose_base_strategy, BaseStrategy};

        match self.config_path.as_ref() {
            Some(path) => Ok(path.clone()),
            None => {
                let strategy = choose_base_strategy()?;
                Ok(strategy.config_dir().join(CONFIG_DIR).join("config.toml"))
            }
        }
    }

    /// Execute command.
    pub async fn execute(&self) -> eyre::Result<()> {
        let config_path = self
            .config_path
            .as_ref()
            .ok_or_eyre("config path is not set")?;
        #[cfg(feature = "remote-wallet")]
        let mut wallet_manager = None;
        let client = if self.command.is_client_required() {
            cfg_if::cfg_if! {
                if #[cfg(feature = "remote-wallet")] {
                    Some(CommandClient::new(&self.config, &mut wallet_manager, self.verbose)?)
                } else {
                    Some(CommandClient::new(&self.config, self.verbose)?)
                }
            }
        } else {
            None
        };
        let store = self.config.store_address();
        self.command
            .execute(Context::new(
                store,
                config_path,
                &self.config,
                client.as_ref(),
                self.verbose,
            ))
            .await
    }
}
