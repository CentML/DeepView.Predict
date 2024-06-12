"""
THIS IS FILE IS INTENDED TO BE USED BY benchmarking.yaml
IT GENERATES A SUMMARY OF CROSS-DEVICE PREDICTIONS IN HTML FORMAT
"""

import argparse
import dominate
from dominate.tags import *
import glob
import pandas as pd
import math
from habitat.analysis import SPECIAL_OPERATIONS

BENCHMARKER_TITLE = "deepview.predict-benchmark"


def get_pct_error_color(pct_err):
    pct_err = abs(pct_err)
    if pct_err < 0.2:
        return "#088567"
    elif 0.2 <= pct_err < 0.4:
        return "#ffa500"
    else:
        return "#ff0000"


def get_pct_err(predicted, measured):
    return round((predicted - measured) / measured, 3)


def generate_summary(e2e_files):
    doc = dominate.document(title=BENCHMARKER_TITLE)

    with doc.head:
        style(
            """\
            body {
             padding-left: 10px;
             margin-bottom: 50px;
             background-color: #F9F8F1;
             color: #2C232A;
             font-family: sans-serif;
            }
            .model-div {
             display: flex;
             flex-direction: row;
             align-items: flex-start;
             gap: 30px;
            }
            """
        )

    for f in sorted(list(glob.glob(f"{e2e_files}/*.csv"))):
        table_tile = f.split("/")[-1].split("-")[0]
        df = pd.read_csv(f)
        devices_names = list(df["origin_device"].unique())
        with doc:
            h1(table_tile)
            with div():
                attr(cls="model-div")
                # end-to-end predictions
                file_table = table()
                table_head = thead()
                header_row = tr()
                header_row += th("org_device")
                header_row += th("dst_device")
                header_row += th("run_time_ms_predicted")
                header_row += th("run_time_ms_measured")
                header_row += th("pct_error")
                table_head += header_row
                file_table.add(table_head)

                table_body = tbody()
                for _, item in df.iterrows():
                    row = tr()
                    row += td(item["origin_device"])
                    row += td(item["dest_device"])
                    row += td(round(item["run_time_ms_predicted"], 3))
                    row += td(round(item["run_time_ms_measured"], 3))
                    row += td(round(item["pct_error"], 3))
                    table_body += row

                file_table.add(table_body)

                # cross prediction table
                cross_pred_table = table()
                table_head = thead()
                header_row = tr()
                header_row += th("from \ to")
                for device in devices_names:
                    header_row += th(f"{device}")
                table_head += header_row
                cross_pred_table.add(table_head)

                # creater NxN table with N = number of devices
                placeholder = [
                    ["x"] * len(devices_names) for _ in range(len(devices_names))
                ]
                hyperlink_names = [
                    ["x"] * len(devices_names) for _ in range(len(devices_names))
                ]
                for _, item in df.iterrows():
                    t_row = devices_names.index(item["origin_device"])
                    t_col = devices_names.index(item["dest_device"])
                    model_bs = table_tile.replace("+", "-")
                    hyperlink_item_name = f"{model_bs}-{item['origin_device']}-{item['dest_device']}-breakdown-combined.html"
                    placeholder[t_row][t_col] = round(item["pct_error"], 3)
                    hyperlink_names[t_row][t_col] = hyperlink_item_name

                table_body = tbody()
                for i, item in enumerate(placeholder):
                    hyperlink_list = hyperlink_names[i]
                    row = tr()
                    row += td(devices_names[i])
                    for j, cross_pred in enumerate(item):
                        if cross_pred == "x":
                            row += td(cross_pred)
                        else:
                            link_name = hyperlink_list[j]
                            color = get_pct_error_color(cross_pred)
                            row += td(
                                a(cross_pred, href=link_name, style=f"color:{color};")
                            )
                    table_body += row
                cross_pred_table.add(table_body)

            footer()

        with open("benchmark_summary.html", "w") as file:
            file.write(doc.render())


def generate_details(ops_files):
    for f in sorted(list(glob.glob(f"{ops_files}/*.csv"))):
        doc = dominate.document(title=BENCHMARKER_TITLE)
        with doc.head:
            style(
                """\
                    body {
                    padding-left: 10px;
                    margin-bottom: 150px;
                    background-color: #F9F8F1;
                    color: #2C232A;
                    font-family: sans-serif;
                    }
                    """
            )

        file_name = f.replace("+", "-").split("/")[-1].replace(".csv", "")
        df = pd.read_csv(f)
        df_special_ops = df[df["operation"].isin(SPECIAL_OPERATIONS)]
        df_no_special_ops = df[~df["operation"].isin(SPECIAL_OPERATIONS)]
        mlp_err = get_pct_err(
            df_special_ops["run_time_ms_predicted"].sum(),
            df_special_ops["run_time_ms_measured"].sum(),
        )
        wave_scale_err = get_pct_err(
            df_no_special_ops["run_time_ms_predicted"].sum(),
            df_no_special_ops["run_time_ms_measured"].sum(),
        )

        err_tbl = [("mlp err", mlp_err), ("wave scale err", wave_scale_err)]

        col_names = df.columns.to_list()
        with doc:
            h1(file_name)
            with div():
                err_table = table()
                table_head = thead()
                header_row = tr()
                header_row += th("category")
                header_row += th("pct err")
                table_head += header_row
                err_table.add(table_head)

                table_body = tbody()
                for name, err in err_tbl:
                    row = tr()
                    row += td(name)
                    row += td(err)
                    table_body += row
                err_table.add(table_body)

            br()

            with div():
                file_table = table()
                table_head = thead()
                header_row = tr()
                for c in col_names:
                    header_row += th(c)
                table_head += header_row
                file_table.add(table_head)

                table_body = tbody()
                for _, item in df.iterrows():
                    row = tr()
                    row += td(item["operation"])
                    row += td(round(item["run_time_ms_predicted"], 3))
                    row += td(round(item["unscaled_predicted_ms"], 3))
                    row += td(round(item["run_time_ms_measured"], 3))
                    row += td(round(item["wgt_pred_time"], 3))
                    row += td(round(item["pct_error"], 3))
                    row += td(
                        item["args"]
                        if isinstance(item["args"], str) or not math.isnan(item["args"])
                        else "[]"
                    )
                    row += td(round(item["ktime_local_ms"], 3))
                    row += td(round(item["runtime_local_ms"], 3))
                    row += td(round(item["predicted_local_ms"], 3))
                    table_body += row

                file_table.add(table_body)

        with open(f"{file_name}.html", "w") as f:
            f.write(doc.render())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", required=True)
    parser.add_argument("--ops", required=True)
    args = parser.parse_args()

    generate_summary(args.e2e)
    generate_details(args.ops)
