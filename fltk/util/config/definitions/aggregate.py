from enum import unique, Enum


@unique
class Aggregations(Enum):
    avg = 'Avg'
    fedavg = 'FedAvg'
    sum = 'Sum'
    median = 'median'
    trmean = 'trmean'
    bulyan = 'bulyan'
    krum = 'krum'
    multiKrum = 'multiKrum'
    krum_pseudo = 'krum_pseudo'
    multiKrum_pseudo = "multiKrum_pseudo"
