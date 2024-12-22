from enum import unique, Enum


@unique
class Aggregations(Enum):
    avg = 'Avg'
    fed_avg = 'fed_avg'
    sum = 'Sum'
    median = 'median'
    trmean = 'trmean'
    bulyan = 'bulyan'
    krum = 'krum'
    multiKrum = 'multiKrum'
    krum_pseudo = 'krum_pseudo'
    multiKrum_pseudo = "multiKrum_pseudo"
    krum_logits = "krum_logits"
    multiKrum_logits = "multiKrum_logits"
    bulyan_logits = "bulyan_logits"
    clustering = "clustering"
