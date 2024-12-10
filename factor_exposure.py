# region imports
from AlgorithmImports import *

import csv
from io import StringIO
# endregion

class NorthFieldFactorExposure(PythonData):

    _drop_unsupported_assets = False  # Change this if you want.

    def get_source(self, config: SubscriptionDataConfig, date: datetime, is_live: bool):
        return SubscriptionDataSource(
            f"US_2_19_9g/RSQRM_US_v2_19_9g{date.strftime('%Y%m%d')}_USDcusip.csv",
            SubscriptionTransportMedium.OBJECT_STORE,
            FileFormat.CSV
        )

    def reader(self, config: SubscriptionDataConfig, line: str, date: datetime, is_live: bool):       
        data = next(csv.reader(StringIO(line)))
        fe = NorthFieldFactorExposure()
        
        # Parse columns.
        fe.name = data[1]
        fe.value = data[3]
        fe.monthly_residual_sd_pct = float(data[4])
        fe.betas = [float(data[i]) for i in range(6, 47)]
        fe.currency_quotation = data[48]
        fe.total_forecast_risk_ann_pct = data[49]

        # QC doesn't have ADRs, OTC stocks, European stocks, Forex without quote currencies, industry proxies...
        if "*" in data[0] or "_" in data[0]:
            fe.symbol = Symbol.create(data[0], SecurityType.BASE, Market.USA)
        else:
            fe.symbol = SecurityDefinitionSymbolResolver.get_instance().cusip(data[0], date)

        if not fe.symbol:
            #NorthFieldFactorExposure.algorithm.log(f"Error creating Symbol for {data[0]} at {date}. Name: {fe.name}")
            # Example CUSIPs that are dropped:
            # 2024-06-05 00:00:00 Error creating Symbol for G9T17Y80 at 2024-06-05 00:00:00. Name: "VANGD.US TRSY.0-1Y (OTC) BD UCITS ETF USD ACC"
            # 2024-06-05 00:00:00 Error creating Symbol for 06254520 at 2024-06-05 00:00:00. Name: "BANK OF HAWAII DRC EACH"
            # 2024-06-05 00:00:00 Error creating Symbol for 57142B10 at 2024-06-05 00:00:00. Name: "MARQETA A"
            # 2024-06-05 00:00:00 Error creating Symbol for G6543112 at 2024-06-05 00:00:00. Name: "NOBLE CORPORATION"
            # 2024-06-05 00:00:00 Error creating Symbol for 87283Q50 at 2024-06-05 00:00:00. Name: "T ROWE PRICE US EQUITY RESEARCH ETF"
            # 2024-06-05 00:00:00 Error creating Symbol for 50076757 at 2024-06-05 00:00:00. Name: "KRANESHARES HANG SENG TECH INDEX ETF"
            # 2024-06-05 00:00:00 Error creating Symbol for 29482Y20 at 2024-06-05 00:00:00. Name: "ERICKSON"
            # 2024-06-05 00:00:00 Error creating Symbol for 44615078 at 2024-06-05 00:00:00. Name: "HUNTINGTON BANCSHARES DEP"
            # 2024-06-05 00:00:00 Error creating Symbol for 41150T20 at 2024-06-05 00:00:00. Name: "HARBOR CUSTOM DEV.8 0 CUM CONV PREF. SR.A"
            if self._drop_unsupported_assets:
                return None
            else:
                fe.symbol = Symbol.create(data[0], SecurityType.BASE, Market.USA)

        fe.end_time = date
        
        return fe