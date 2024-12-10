# region imports
from AlgorithmImports import *
# endregion


class NorthFieldAlpha(PythonData):

    def get_source(self, config: SubscriptionDataConfig, date: datetime, is_live: bool):
        return SubscriptionDataSource(
            f"Alphas/Alphas {date.strftime('%Y%m%d')}.txt",
            SubscriptionTransportMedium.OBJECT_STORE,
            FileFormat.CSV
        )

    def reader(self, config: SubscriptionDataConfig, line: str, date: datetime, is_live: bool):
        if not line[0].isdigit():
            return None
        data = line.split('|')
        asset = NorthFieldAlpha()
        asset.symbol = SecurityDefinitionSymbolResolver.get_instance().cusip(data[0], date)
        if not asset.symbol:
            return
        asset.name = data[1]
        try:
            asset.alpha = float(data[2]) # These fields can be empty in the data file.
            asset.market_cap = float(data[3])
        except:
            return None
        asset.end_time = date #+ timedelta(1)
        return asset
