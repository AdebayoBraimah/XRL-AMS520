# region imports
from AlgorithmImports import *
# endregion

class NorthFieldFactorDefinition(PythonData):

    def get_source(self, config: SubscriptionDataConfig, date: datetime, is_live: bool):
        return SubscriptionDataSource(
            f"US_2_19_9g/FF_RSQ_RSQRM_US_v2_19_9g_USD_{date.strftime('%Y%m%d')}_FactorDef.txt",
            SubscriptionTransportMedium.OBJECT_STORE,
            FileFormat.CSV
        )

    def reader(self, config: SubscriptionDataConfig, line: str, date: datetime, is_live: bool):
        if not line[0].isdigit():
            return None
        data = line.split('|')
        fd = NorthFieldFactorDefinition()
        
        # Parse columns.
        fd.name = data[1]
        fd.code = data[2]
        fd.variance = float(data[3])
        fd.std_dev_residuals = math.sqrt(fd.variance)

        fd.symbol = Symbol.create(fd.code, SecurityType.BASE, Market.USA, fd.name)
        fd.end_time = date

        return fd