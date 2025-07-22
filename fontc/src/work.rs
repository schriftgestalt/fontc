//! Helps fontc manage workloads that span FE and BE.
//!
//! Basically enums that can be a FeWhatever or a BeWhatever.

use fontbe::orchestration::{AnyWorkId, BeWork, Context as BeContext};
use fontdrasil::orchestration::{Access, AccessType};
use fontir::orchestration::{Context as FeContext, IrWork, WorkId};

use crate::Error;

// Work of any type, FE, BE, ... some future pass, w/e
#[derive(Debug)]
pub enum AnyWork {
    Fe(Box<IrWork>),
    Be(Box<BeWork>),
    /// Used to get things marked completed w/o actually running. Skipping basically.
    Nop(AnyWorkId, AnyAccess),
    /// Used for work that isn't actually run, just gets marked complete on completion of some other work
    AlsoComplete(AnyWorkId, AnyAccess),
}

impl From<Box<IrWork>> for AnyWork {
    fn from(work: Box<IrWork>) -> Self {
        AnyWork::Fe(work)
    }
}

impl From<Box<BeWork>> for AnyWork {
    fn from(work: Box<BeWork>) -> Self {
        AnyWork::Be(work)
    }
}

impl AnyWork {
    pub fn id(&self) -> AnyWorkId {
        match self {
            AnyWork::Be(work) => work.id(),
            AnyWork::Fe(work) => work.id().into(),
            AnyWork::Nop(id, ..) | AnyWork::AlsoComplete(id, ..) => id.clone(),
        }
    }

    pub fn read_access(&self) -> AnyAccess {
        match self {
            AnyWork::Be(work) => work.read_access().into(),
            AnyWork::Fe(work) => work.read_access().into(),
            AnyWork::Nop(.., access) | AnyWork::AlsoComplete(.., access) => access.clone(),
        }
    }

    pub fn write_access(&self) -> AnyAccess {
        match self {
            AnyWork::Be(work) => work.write_access().into(),
            AnyWork::Fe(work) => work.write_access().into(),
            AnyWork::Nop(..) | AnyWork::AlsoComplete(..) => AnyAccess::Fe(Access::None),
        }
    }

    pub fn also_completes(&self) -> Vec<AnyWorkId> {
        match self {
            AnyWork::Be(work) => work.also_completes().into_iter().collect(),
            AnyWork::Fe(work) => work
                .also_completes()
                .into_iter()
                .map(|id| id.into())
                .collect(),
            AnyWork::Nop(..) | AnyWork::AlsoComplete(..) => Vec::new(),
        }
    }

    pub fn exec(&self, context: AnyContext) -> Result<(), Error> {
        match self {
            AnyWork::Be(work) => work.exec(context.unwrap_be()).map_err(|e| e.into()),
            AnyWork::Fe(work) => work.exec(context.unwrap_fe()).map_err(|e| e.into()),
            AnyWork::Nop(..) | AnyWork::AlsoComplete(..) => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnyAccess {
    Be(Access<AnyWorkId>),
    Fe(Access<WorkId>),
}

impl From<Access<AnyWorkId>> for AnyAccess {
    fn from(value: Access<AnyWorkId>) -> Self {
        AnyAccess::Be(value)
    }
}

impl From<Access<WorkId>> for AnyAccess {
    fn from(value: Access<WorkId>) -> Self {
        AnyAccess::Fe(value)
    }
}

impl AnyAccess {
    pub fn check(&self, id: &AnyWorkId) -> bool {
        match self {
            AnyAccess::Be(access) => access.check(id),
            AnyAccess::Fe(access) => {
                let AnyWorkId::Fe(id) = id else {
                    return false;
                };
                access.check(id)
            }
        }
    }

    pub fn to_be(&self) -> Access<AnyWorkId> {
        match self {
            AnyAccess::Fe(access) => match access {
                Access::All => Access::All,
                Access::None => Access::None,
                Access::Unknown => Access::Unknown,
                Access::Variant(id) => Access::Variant(id.clone().into()),
                Access::SpecificInstanceOfVariant(id) => {
                    Access::SpecificInstanceOfVariant(id.clone().into())
                }
                Access::Set(ids) => Access::Set(
                    ids.iter()
                        .map(|id| match id {
                            AccessType::SpecificInstanceOfVariant(id) => {
                                AccessType::SpecificInstanceOfVariant(id.clone().into())
                            }
                            AccessType::Variant(exemplar) => {
                                AccessType::Variant(exemplar.clone().into())
                            }
                        })
                        .collect(),
                ),
            },
            AnyAccess::Be(access) => access.clone(),
        }
    }

    pub fn to_fe(&self) -> Access<WorkId> {
        match self {
            AnyAccess::Fe(access) => access.clone(),
            AnyAccess::Be(access) => match access {
                Access::All => Access::All,
                Access::None => Access::None,
                Access::Unknown => Access::Unknown,
                Access::Variant(id) => match id {
                    AnyWorkId::Fe(id) => Access::Variant(id.clone()),
                    AnyWorkId::Be(..) | AnyWorkId::InternalTiming(..) => Access::None,
                },
                Access::SpecificInstanceOfVariant(id) => match id {
                    AnyWorkId::Fe(id) => Access::SpecificInstanceOfVariant(id.clone()),
                    AnyWorkId::Be(..) | AnyWorkId::InternalTiming(..) => Access::None,
                },
                Access::Set(ids) => Access::Set(
                    ids.iter()
                        .filter_map(|id| match id {
                            AccessType::SpecificInstanceOfVariant(id) => match id {
                                AnyWorkId::Fe(id) => {
                                    Some(AccessType::SpecificInstanceOfVariant(id.clone()))
                                }
                                AnyWorkId::Be(..) | AnyWorkId::InternalTiming(..) => None,
                            },
                            AccessType::Variant(exemplar) => match exemplar {
                                AnyWorkId::Fe(id) => Some(AccessType::Variant(id.clone())),
                                AnyWorkId::Be(..) | AnyWorkId::InternalTiming(..) => None,
                            },
                        })
                        .collect(),
                ),
            },
        }
    }
}

pub enum AnyContext {
    Fe(Box<FeContext>),
    Be(Box<BeContext>),
}

impl AnyContext {
    pub fn for_work(
        fe_root: &FeContext,
        be_root: &BeContext,
        work_id: &AnyWorkId,
        read_access: AnyAccess,
        write_access: AnyAccess,
    ) -> AnyContext {
        match work_id {
            AnyWorkId::Be(..) => AnyContext::Be(
                be_root
                    .copy_for_work(read_access.to_be(), write_access.to_be())
                    .into(),
            ),
            AnyWorkId::Fe(..) => AnyContext::Fe(
                fe_root
                    .copy_for_work(read_access.to_fe(), write_access.to_fe())
                    .into(),
            ),
            AnyWorkId::InternalTiming(..) => {
                panic!("Should never create a context for internal timing")
            }
        }
    }

    pub fn unwrap_be(&self) -> &BeContext {
        match self {
            AnyContext::Fe(..) => panic!("Not a BE context"),
            AnyContext::Be(ctx) => ctx,
        }
    }

    pub fn unwrap_fe(&self) -> &FeContext {
        match self {
            AnyContext::Fe(ctx) => ctx,
            AnyContext::Be(..) => panic!("Not a FE context"),
        }
    }
}
