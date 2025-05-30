use anchor_lang::{prelude::*, ZeroCopy};

use crate::{
    events::EventEmitter,
    states::{
        common::action::{Action, ActionParams, Closable},
        NonceBytes, StoreWalletSigner,
    },
    CoreError,
};

use super::Authenticate;

/// Create Action.
pub(crate) trait Create<'info, A>: Sized + anchor_lang::Bumps {
    /// Create Params.
    type CreateParams: ActionParams;

    /// Get the action account.
    fn action(&self) -> AccountInfo<'info>;

    /// Get the payer account.
    fn payer(&self) -> AccountInfo<'info>;

    /// Get the seeds of the payer.
    fn payer_seeds(&self) -> Result<Option<Vec<Vec<u8>>>> {
        Ok(None)
    }

    /// Get the system program account.
    fn system_program(&self) -> AccountInfo<'info>;

    /// Validate.
    fn validate(&self, _params: &Self::CreateParams) -> Result<()>;

    /// The implementation of the creation.
    fn create_impl(
        &mut self,
        params: &Self::CreateParams,
        nonce: &NonceBytes,
        bumps: &Self::Bumps,
        remaining_accounts: &'info [AccountInfo<'info>],
        callback_version: Option<u8>,
    ) -> Result<()>;

    /// Create Action.
    fn create(
        ctx: &mut Context<'_, '_, 'info, 'info, Self>,
        nonce: &NonceBytes,
        params: &Self::CreateParams,
        callback_version: Option<u8>,
    ) -> Result<()> {
        let accounts = &mut ctx.accounts;
        accounts.validate(params)?;
        accounts.transfer_execution_lamports(params)?;
        accounts.create_impl(
            params,
            nonce,
            &ctx.bumps,
            ctx.remaining_accounts,
            callback_version,
        )?;
        Ok(())
    }

    /// Transfer execution lamports.
    fn transfer_execution_lamports(&self, params: &Self::CreateParams) -> Result<()> {
        use crate::ops::execution_fee::TransferExecutionFeeOperation;

        let payer_seeds = self.payer_seeds()?;
        let payer_seeds = payer_seeds
            .as_ref()
            .map(|seeds| seeds.iter().map(|seed| seed.as_slice()).collect::<Vec<_>>());

        TransferExecutionFeeOperation::builder()
            .payment(self.action())
            .payer(self.payer())
            .execution_lamports(params.execution_lamports())
            .system_program(self.system_program())
            .signer_seeds(payer_seeds.as_deref())
            .build()
            .execute()
    }
}

type IsCallerOwner = bool;
pub(crate) type Success = bool;

/// Close Action.
pub(crate) trait Close<'info, A>: Authenticate<'info>
where
    A: Action + ZeroCopy + Owner + Closable,
{
    /// Expected keeper role.
    fn expected_keeper_role(&self) -> &str;

    /// Rent receiver.
    fn rent_receiver(&self) -> AccountInfo<'info>;

    /// Get event authority.
    fn event_authority(&self, bumps: &Self::Bumps) -> (AccountInfo<'info>, u8);

    /// Get store wallet bump.
    fn store_wallet_bump(&self, bumps: &Self::Bumps) -> u8;

    /// Whether to skip the completion check when the authority is keeper.
    fn skip_completion_check_for_keeper(&self) -> Result<bool> {
        Ok(false)
    }

    /// Validate.
    fn validate(&self) -> Result<()>;

    /// Process before the close.
    fn process(
        &self,
        is_caller_owner: bool,
        store_wallet_signer: &StoreWalletSigner,
        event_emitter: &EventEmitter<'_, 'info>,
    ) -> Result<Success>;

    /// Close Action.
    fn close(ctx: &Context<'_, '_, '_, 'info, Self>, reason: &str) -> Result<()> {
        let accounts = &ctx.accounts;
        accounts.validate()?;
        let is_caller_owner = accounts.preprocess()?;

        let store_wallet_signer = StoreWalletSigner::new(
            accounts.store().key(),
            accounts.store_wallet_bump(&ctx.bumps),
        );

        let (authority, bump) = accounts.event_authority(&ctx.bumps);
        let event_emitter = EventEmitter::new(&authority, bump);

        if accounts.process(is_caller_owner, &store_wallet_signer, &event_emitter)? {
            {
                let action_address = accounts.action().key();
                let action = accounts.action().load()?;
                let event = action.to_closed_event(&action_address, reason)?;
                event_emitter.emit_cpi(&event)?;
            }
            accounts.close_action_account()?;
        } else {
            msg!("Some ATAs are not initialized, skip the close");
        }
        Ok(())
    }

    /// Action.
    fn action(&self) -> &AccountLoader<'info, A>;

    /// Preprocess.
    fn preprocess(&self) -> Result<IsCallerOwner> {
        if *self.authority().key == self.action().load()?.header().owner {
            Ok(true)
        } else {
            self.only_role(self.expected_keeper_role())?;
            {
                let action = self.action().load()?;
                if self.skip_completion_check_for_keeper()?
                    || action.header().action_state()?.is_completed_or_cancelled()
                {
                    Ok(false)
                } else {
                    err!(CoreError::PermissionDenied)
                }
            }
        }
    }

    /// Close the action account.
    fn close_action_account(&self) -> Result<()> {
        self.action().close(self.rent_receiver())
    }
}
