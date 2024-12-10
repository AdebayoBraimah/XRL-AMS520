# region imports
from AlgorithmImports import *
# endregion


class NorthFieldUniverse(PythonData):

    def get_source(self, config: SubscriptionDataConfig, date: datetime, is_live: bool):
        return SubscriptionDataSource(
            f"GeneralUniverse/Universe {date.strftime('%Y%m%d')}.txt",
            SubscriptionTransportMedium.OBJECT_STORE,
            FileFormat.CSV
        )

    def reader(self, config: SubscriptionDataConfig, line: str, date: datetime, is_live: bool):
        if not line[0].isdigit():
            return None
        data = line.split('|')
        asset = NorthFieldUniverse()
        symbol = SecurityDefinitionSymbolResolver.get_instance().cusip(data[0], date)
        if not symbol:
            return
        asset.symbol = symbol
        asset.market_cap = float(data[1])
        asset.name = data[2]
        asset.value = asset.market_cap
        asset.end_time = date
        return asset


class NorthFieldInvestableUniverse(NorthFieldUniverse):

    def get_source(self, config: SubscriptionDataConfig, date: datetime, is_live: bool):
        return SubscriptionDataSource(
            f"InvestableUniverse/InvestableUniverse {date.strftime('%Y%m%d')}.txt",
            SubscriptionTransportMedium.OBJECT_STORE,
            FileFormat.CSV
        )