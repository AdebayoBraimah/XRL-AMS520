# region imports
from AlgorithmImports import *
# endregion

class NorthFieldFactorCorrelation(PythonData):

    _rows_seen_by_date = {}

    def get_source(self, config: SubscriptionDataConfig, date: datetime, is_live: bool):
        return SubscriptionDataSource(
            f"US_2_19_9g/FF_RSQ_RSQRM_US_v2_19_9g_USD_{date.strftime('%Y%m%d')}_Correl.txt",
            SubscriptionTransportMedium.OBJECT_STORE,
            FileFormat.CSV
        )

    def reader(self, config: SubscriptionDataConfig, line: str, date: datetime, is_live: bool):
        data = line.split('|')
        try:
            float(data[0])
        except:
            self._columns = data
            return None

        if date not in self._rows_seen_by_date:
            self._rows_seen_by_date[date] = 0
        else:
            self._rows_seen_by_date[date] += 1

        fc = NorthFieldFactorCorrelation()
        factor_name = self._columns[self._rows_seen_by_date[date]].replace(' ', '')
        fc.symbol = Symbol.create(factor_name, SecurityType.BASE, Market.USA)
        fc.end_time = date
        
        # Parse columns.
        fc.series = pd.Series(data, index=self._columns).astype(float)

        return fc
