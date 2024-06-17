import json
import logging
from typing import List

import pudb
from langchain_core.documents import Document

from load import Loader

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


class CustomLoader(Loader):

    def load_file(self, file_path: str) -> List[Document]:
        # TODO(STP): Convert pseudocode.
        file_extension = file_path.split(".")[-1]
        if file_extension == "json":
            with open(file_path) as fin:
                data = json.load(fin)
                text = data["text"]
                del data["text"]
                # TODO(STP): Add the filename to the metadata.
                metadata = data
                return [Document(page_content=text, metadata=metadata)]
        else:
            super().load_file(file_path)


# pudb.set_trace()

if __name__ == "__main__":
    print("hello")
    logging.debug("Hello - logging")
    # loader = CustomLoader(
    #     "data/financial_dataset.zip",
    #     is_zipped=True,
    #     output_location="foo",
    #     num_workers=10,
    # )

    loader = CustomLoader(
        "data/financial_dataset",
        is_zipped=False,
        output_location="foo_regular",
        num_workers=10,
    )

    loader.load_dataset()
