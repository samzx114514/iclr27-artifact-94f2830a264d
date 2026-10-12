from .base import UpdateDomainContract
from .code import CONTRACT as CODE
from .math import CONTRACT as MATH
from .research import CONTRACT as RESEARCH
from .search import CONTRACT as SEARCH
from .sql import CONTRACT as SQL

_CONTRACTS = {
    item.domain: item for item in (MATH, SQL, SEARCH, RESEARCH, CODE)
}


def get_update_domain_contract(domain: str) -> UpdateDomainContract:
    try:
        return _CONTRACTS[domain]
    except KeyError as error:
        raise ValueError(f"unsupported update domain: {domain}") from error


__all__ = ["UpdateDomainContract", "get_update_domain_contract"]
