import os

from kartverkets_api import kartverketsAPI
from sibr_module import BigQuery,Logger
import argparse
import asyncio
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

print(f'USERNAME: {os.getenv("GRUNNBOK_USERNAME")}')

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument("--by-properties", nargs="*", help="gets information about a specific property based on kommunenr, gnr, bnr, festnr & seksjonsnr")
group.add_argument("--by-period", action="store_true", help="gets information about all properties within a given period")
group.add_argument("-up","--update-project", action="store_true", help="gets information about all properties within a given period")

parser.add_argument("--start-date", help="The start date for the period (required with --get_by_period)")
parser.add_argument("--end-date", help="The end date for the period (required with --get_by_period)")
parser.add_argument("--transfer-type",default="active",help="Transfer type; Choose between active or historical transfers")
parser.add_argument("-s","--save", action="store_true", help="A boolean for storing data to Big Query")
parser.add_argument("-ost","--ownership-type", type = str, choices=["eier","andel"])


test_properties = properties = [
    {
        "kommunenummer": "3212",
        "gaardsnummer": 1,
        "bruksnummer": 2,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 1,
        "bruksnummer": 5,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 1,
        "bruksnummer": 9,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 1,
        "bruksnummer": 12,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 120,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 120,
        "festenummer": 1,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 120,
        "festenummer": 2,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 120,
        "festenummer": 3,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 125,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 125,
        "festenummer": 0,
        "seksjonsnummer": 1
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 125,
        "festenummer": 0,
        "seksjonsnummer": 2
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 125,
        "festenummer": 0,
        "seksjonsnummer": 3
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 125,
        "festenummer": 0,
        "seksjonsnummer": 4
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 150,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 151,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 152,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 153,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 253,
        "festenummer": 0,
        "seksjonsnummer": 1
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 253,
        "festenummer": 0,
        "seksjonsnummer": 2
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 253,
        "festenummer": 0,
        "seksjonsnummer": 3
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 253,
        "festenummer": 0,
        "seksjonsnummer": 4
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 253,
        "festenummer": 0,
        "seksjonsnummer": 5
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 253,
        "festenummer": 0,
        "seksjonsnummer": 6
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 253,
        "festenummer": 0,
        "seksjonsnummer": 7
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 253,
        "festenummer": 0,
        "seksjonsnummer": 8
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 410,
        "festenummer": 0,
        "seksjonsnummer": 1
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 410,
        "festenummer": 0,
        "seksjonsnummer": 2
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 410,
        "festenummer": 0,
        "seksjonsnummer": 3
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 410,
        "festenummer": 0,
        "seksjonsnummer": 4
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 861,
        "festenummer": 0,
        "seksjonsnummer": 1
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 2,
        "bruksnummer": 861,
        "festenummer": 0,
        "seksjonsnummer": 2
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 4,
        "bruksnummer": 318,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 4,
        "bruksnummer": 318,
        "festenummer": 1,
        "seksjonsnummer": 1
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 4,
        "bruksnummer": 318,
        "festenummer": 1,
        "seksjonsnummer": 2
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 4,
        "bruksnummer": 318,
        "festenummer": 1,
        "seksjonsnummer": 3
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 4,
        "bruksnummer": 485,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 4,
        "bruksnummer": 485,
        "festenummer": 0,
        "seksjonsnummer": 1
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 4,
        "bruksnummer": 485,
        "festenummer": 0,
        "seksjonsnummer": 2
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 6,
        "bruksnummer": 189,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 6,
        "bruksnummer": 190,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 6,
        "bruksnummer": 191,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 6,
        "bruksnummer": 192,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 9,
        "bruksnummer": 88,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 9,
        "bruksnummer": 88,
        "festenummer": 1,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 9,
        "bruksnummer": 88,
        "festenummer": 2,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 9,
        "bruksnummer": 88,
        "festenummer": 3,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 10,
        "bruksnummer": 53,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 28,
        "bruksnummer": 324,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 28,
        "bruksnummer": 888,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 29,
        "bruksnummer": 14,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 29,
        "bruksnummer": 87,
        "festenummer": 0,
        "seksjonsnummer": 1
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 29,
        "bruksnummer": 87,
        "festenummer": 0,
        "seksjonsnummer": 2
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 29,
        "bruksnummer": 87,
        "festenummer": 0,
        "seksjonsnummer": 3
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 30,
        "bruksnummer": 1,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 30,
        "bruksnummer": 3,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 30,
        "bruksnummer": 4,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 30,
        "bruksnummer": 5,
        "festenummer": 0,
        "seksjonsnummer": 0
    },
    {
        "kommunenummer": "3212",
        "gaardsnummer": 30,
        "bruksnummer": 6,
        "festenummer": 0,
        "seksjonsnummer": 0
    }
]

def transform_cadastrals(dataframe : pd.DataFrame,request_cols) -> pd.DataFrame:
    df = dataframe.copy()
    rename = {"municipality_num": "kommunenummer",
              "cadastral_num": "gaardsnummer",
              "unit_num": "bruksnummer",
              "leasehold_num": "festenummer",
              "section_num": "seksjonsnummer"}
    df.rename(columns=rename, inplace=True)
    if "item_id" in df.columns:
        df.set_index("item_id",inplace=True)
    else:
        print(f'WARNING: item_id not found in df')
    df.dropna(subset=["gaardsnummer", "bruksnummer","price"], inplace=True)
    for col in ["festenummer","seksjonsnummer"]:
        df[col].fillna(0, inplace=True)
    df = df.loc[df["gaardsnummer"] != 0, :]
    #df["last_updated"] = pd.to_datetime(df["last_updated"], utc=True)
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    #df.loc[:,"scrape_date"] = pd.to_datetime(df["scrape_date"])

    for col in request_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            #df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f'Column {col} not found in df')
    df.dropna(subset=request_cols, inplace=True)
    for col in request_cols:
        if col in df.columns:
            #df.loc[:,col] = df[col].astype(int)
            df[col] = df[col].astype(int)
            if col == "kommunenummer":
                #df.loc[:, col] = df[col].apply(lambda x: str(x).zfill(4))
                df[col] = df[col].apply(lambda x: str(x).zfill(4))
        else:
            print(f'Column {col} not found in df')
    df = df.loc[df["kommunenummer"] != "0000", :]
    df.sort_values(by=["scrape_date"], inplace=True, ascending=False)
    df.drop_duplicates(subset=request_cols,keep = "first",inplace=True)
    return df
def transform_coop(dataframe : pd.DataFrame,request_cols) -> pd.DataFrame:
    df = dataframe.copy()
    rename = {"coop_org_num": "borettslagnummer",
              "coop_unit_num": "andelsnummer",
              }
    df.rename(columns=rename, inplace=True)
    if "item_id" in df.columns:
        df.set_index("item_id",inplace=True)
    else:
        print(f'WARNING: item_id not found in df')
    df.dropna(subset=["borettslagnummer","andelsnummer","price"], inplace=True)
    df["andelsnummer"].fillna(0, inplace=True)
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])

    for col in request_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f'Column {col} not found in df')
    df.dropna(subset=request_cols, inplace=True)
    for col in request_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
        else:
            print(f'Column {col} not found in df')
    df.sort_values(by=["scrape_date"], inplace=True, ascending=False)
    df.drop_duplicates(subset=request_cols,keep = "first",inplace=True)
    return df


async def main():
    logger = Logger("kartverketMain")
    api = kartverketsAPI(logger=logger)
    bq = BigQuery(logger=logger,project_id="sibr-market")

    args = parser.parse_args()

    try:
        if args.by_properties:
            try:
                df = await api.get_by_property(args.get_by_properties, transfer_type=args.transfer_type)
                if args.save_bq:
                    trouble_columns = [
                        #  "oppdateringsdato_timestamp",
                        # "endretavids_cachedvalue_item",
                        "omsetning_omsatteregisterenhetsretter_item"
                    ]
                    for col in trouble_columns:
                        df[col] = df[col].astype(str)
                    bq.to_bq(df=df,
                             dataset_name = "admin",
                             table_name = "kartverket")
            except Exception as e:
                logger.error(f'Error getting properties: {e}')

        elif args.by_period:
            if args.start_date and args.end_date:
                api.get_by_period(args.start_date, args.end_date)

        elif args.update_project:
            ownership_type_arg = args.ownership_type
            if not ownership_type_arg:
                for ownership_type in ["eier", "andel"]:
                    if ownership_type == "eier":
                        query = """
                            SELECT
                              item_id,
                              last_updated,
                              scrape_date,
                              price,
                              municipality_num,
                              cadastral_num,
                              unit_num,
                              leasehold_num,
                              section_num
                            FROM `sibr-market.clean.homes` h
                              WHERE NOT EXISTS (SELECT 1 FROM staging.cadastrals c WHERE c.item_id = h.item_id) 
                              AND cadastral_num IS NOT NULL
                              AND unit_num IS NOT NULL
                              AND DATE(scrape_date) !=  (SELECT MAX(scrape_date) FROM raw.homes)
                              AND LOWER(ownership_type) = 'eier' OR LOWER(ownership_type) = 'eier '
                            """
                        request_cols = ["kommunenummer", "gaardsnummer", "bruksnummer", "festenummer", "seksjonsnummer"]
                    elif ownership_type == "andel":
                        request_cols = ["borettslagnummer", "andelsnummer"]
                        query = """
                                SELECT item_id, 
                                       last_updated, 
                                       scrape_date, 
                                       price, 
                                       coop_unit_num, 
                                       coop_org_num
                                FROM `sibr-market.clean.homes` h
                                WHERE NOT EXISTS (SELECT 1 FROM staging.cadastrals c WHERE c.item_id = h.item_id)
                                  AND cadastral_num IS NOT NULL
                                  AND unit_num IS NOT NULL
                                  AND DATE (scrape_date) != (SELECT MAX (scrape_date) FROM raw.homes)
                                  AND LOWER (ownership_type) = 'andel' 
                                """
                    else:
                        raise TypeError(f'Expected "eier" or "andel" but got {ownership_type}')
                    db = bq.read_bq(query)

                    #TRANSFORM DATA FOR KARTVERKET

                    db = transform_cadastrals(db, request_cols=request_cols) if ownership_type == "eier" else transform_coop(db,request_cols=request_cols)
                    # if ownership_type == "eier":
                    #     db = transform_cadastrals(db,request_cols=request_cols)
                    # elif ownership_type == "andel":
                    #     db = transform_coop(db,request_cols=request_cols)
                    properties = db[request_cols].to_dict(orient='records')
                    logger.info(f'Total batch size is {len(properties)}')
                    for batch in range(0,len(properties),20000):
                        logger.info(f'Working batch with {len(properties[batch:batch+20000])} of {len(properties)} properties ({len(properties[batch:batch+20000]) / len(properties) * 100:.2f}% of total properties)')
                        try:
                            df = await api.get_by_property(properties[batch:batch+20000], transfer_type=args.transfer_type, ownership_type=ownership_type)
                        except Exception as e:
                            logger.error(f'Error getting properties: {e}')
                            raise

                        logger.info(f'➡️  Input from batch kartverket API: {len(df)}')
                        for col in request_cols:
                            if df[col].dtype != db[col].dtype:
                                logger.warning(f'Column {col} has different data types in db and df. `df` has dtype {df[col].dtype} and `db` has dtype {db[col].dtype}. Forcing both to int')
                                df[col] = df[col].astype(int)
                                db[col] = db[col].astype(int)
                        m = pd.merge(df, db.reset_index(), on=request_cols, how="left")
                        view = m.loc[(m["registreringstidspunkt"] >= "2024-07-01"), ["item_id", "scrape_date",
                                                                                     "registreringstidspunkt",
                                                                                     "omsetning_vederlag_beloepsverdi", "price"]]
                        view = view[view["omsetning_vederlag_beloepsverdi"] > 0]
                        view["dt_diff"] = (view["registreringstidspunkt"] - view["scrape_date"]).dt.days
                        view["price_diff"] = view["omsetning_vederlag_beloepsverdi"] - view["price"]
                        minmax = 3000000
                        view = view[(view['price_diff'] < minmax) & (view['price_diff'] > -minmax)]
                        view = view[(view["dt_diff"]>0) & (view["dt_diff"]<300)]
                        view.rename(columns={"omsetning_vederlag_beloepsverdi": "sale_price"}, inplace=True)
                        logger.info(f'⬅️    Output from batch kartverket API: {len(view)}')
                        if args.save:
                            bq.to_bq(df=view, table_name="cadastrals", dataset_name="staging", if_exists="merge",merge_on=["item_id"])


    except Exception as e:
        logger.error(f'Error getting properties: {e}')

    finally:
        await api.close()


if __name__ == "__main__" :
    asyncio.run(main())