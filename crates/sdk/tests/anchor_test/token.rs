use gmsol_sdk::{client::ops::ExchangeOps, constants::MARKET_USD_UNIT, ops::TokenAccountOps};
use tracing::Instrument;

use crate::anchor_test::setup::{current_deployment, Deployment};

#[tokio::test]
async fn unwrap_native_token_with_swap_path() -> eyre::Result<()> {
    let deployment = current_deployment().await?;
    let _guard = deployment.use_accounts().await?;
    let span = tracing::info_span!("unwrap_native_token_with_swap_path");
    let _enter = span.enter();

    let long_token_amount = 1120 * 1_000_000 / 10_000;
    let long_swap_token_amount = 1130 * 1_000_000_000 / 100;
    let short_token_amount = 2340 * 100_000_000;
    let market_token = deployment
        .prepare_market(
            ["SOL", "fBTC", "USDG"],
            long_token_amount,
            short_token_amount,
            true,
        )
        .await?;
    let swap_market_token = deployment
        .prepare_market(
            ["fBTC", "WSOL", "USDG"],
            long_swap_token_amount,
            short_token_amount,
            true,
        )
        .await?;
    let wsol = deployment.token("WSOL").unwrap();

    let store = &deployment.store;
    let oracle = &deployment.oracle();
    let client = deployment.user_client(Deployment::DEFAULT_USER)?;
    let keeper = deployment.user_client(Deployment::DEFAULT_KEEPER)?;

    let collateral_amount = 210 * 100_000_000;
    let size = 40 * MARKET_USD_UNIT;
    deployment
        .mint_or_transfer_to_user("USDG", Deployment::DEFAULT_USER, collateral_amount)
        .await?;

    // Open position.
    let (rpc, order) = client
        .market_increase(store, market_token, false, collateral_amount, true, size)
        .build_with_address()
        .await?;
    let signature = rpc.send().await?;
    tracing::info!(%order, %signature, %size, "created an order to increase position");

    let mut builder = keeper.execute_order(store, oracle, &order, false)?;
    deployment
        .execute_with_pyth(
            builder
                .add_alt(deployment.common_alt().clone())
                .add_alt(deployment.market_alt().clone()),
            None,
            true,
            true,
        )
        .instrument(tracing::info_span!("execute", order=%order))
        .await?;

    // Close position.
    let receiver = keeper.payer();
    let (rpc, order) = client
        .market_decrease(store, market_token, false, collateral_amount, true, size)
        .final_output_token(&wsol.address)
        .swap_path(vec![*swap_market_token])
        .receiver(receiver)
        .build_with_address()
        .await?;
    let signature = rpc.send().await?;
    tracing::info!(%order, %signature, %size, %receiver, "created an order to close position");

    let mut builder = keeper.execute_order(store, oracle, &order, false)?;
    deployment
        .execute_with_pyth(
            builder
                .add_alt(deployment.common_alt().clone())
                .add_alt(deployment.market_alt().clone()),
            None,
            true,
            true,
        )
        .instrument(tracing::info_span!("execute", order=%order))
        .await?;

    Ok(())
}

#[tokio::test]
async fn token_metadata() -> eyre::Result<()> {
    let deployment = current_deployment().await?;
    let _guard = deployment.use_accounts().await?;
    let span = tracing::info_span!("token_metadata");
    let _enter = span.enter();

    let client = deployment.user_client(Deployment::DEFAULT_KEEPER)?;
    let store = &deployment.store;

    let market_token = deployment
        .market_token("SOL", "fBTC", "USDG")
        .expect("must exist");

    let glv_token = &deployment.glv_token;

    // Create metadata for a market token.
    let (rpc, metadata) = client
        .create_token_metadata(
            store,
            market_token,
            "Market Token 1".to_string(),
            "GM1".to_string(),
            "metadata-uri".to_string(),
        )
        .swap_output(());
    tracing::info!(%market_token, "creating token metadata: {metadata}");
    let signature = rpc.send_without_preflight().await?;

    tracing::info!(%signature, %market_token, "created token metadata: {metadata}");

    let signature = client
        .update_token_metadata_by_mint(
            store,
            market_token,
            "Market Token 1".to_string(),
            "GM1".to_string(),
            "metadata-uri-2".to_string(),
        )
        .send_without_preflight()
        .await?;

    tracing::info!(%signature, %market_token, "updated token metadata: {metadata}");

    // Create metadata for a glv token.
    let (rpc, metadata) = client
        .create_token_metadata(
            store,
            glv_token,
            "GLV Token 1".to_string(),
            "GLV1".to_string(),
            "metadata-uri".to_string(),
        )
        .swap_output(());
    tracing::info!(%glv_token, "creating token metadata: {metadata}");
    let signature = rpc.send_without_preflight().await?;

    tracing::info!(%signature, %glv_token, "created token metadata: {metadata}");

    let signature = client
        .update_token_metadata_by_mint(
            store,
            glv_token,
            "GLV Token 1".to_string(),
            "GLV1".to_string(),
            "metadata-uri-2".to_string(),
        )
        .send_without_preflight()
        .await?;

    tracing::info!(%signature, %glv_token, "updated token metadata: {metadata}");

    Ok(())
}
