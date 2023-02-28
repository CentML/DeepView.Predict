
import argparse
import dominate
from dominate.tags import *
import os
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("title", type=str)
    parser.add_argument('-f', '--file-list', nargs='+', default=[])

    args = parser.parse_args()

    doc_title = args.title
    file_list = args.file_list
    doc = dominate.document(title=doc_title)

    with doc:
        h1(doc_title)
        file_table = table()
        table_head = thead()
        header_row = tr()
        header_row += th("File name")
        header_row += th("Date Updated")
        table_head += header_row
        file_table.add(table_head)

        table_body = tbody()
        for file in file_list:
            if os.path.exists(file):
                row = tr()
                row += td(a(file, href=file))
                row += td(datetime.fromtimestamp(os.path.getmtime(file)).isoformat())
                table_body += row
        file_table.add(table_body)
    with open("index.html", "w") as file:
        file.write(doc.render())