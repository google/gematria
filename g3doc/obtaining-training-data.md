# Obtaining Training Data

This document gives an overview of how to use the Gematria tooling to
parse/process training data from multiple sources.

## Parsing training data from the BHive dataset

The publically available [BHive](https://github.com/ithemal/bhive) dataset has
about 300k basic blocks along with throughput data for Ivy Bridge, Haswell, and
Skylake in a CSV format. Gematria provides native tooling to process these CSV
files into `.tfrecord` files that can be used with Gematria models.

Every line in the BHive CSV files has the format `<machine code of BB in
hex>,<throughput in cycles>`. This is referred to as the BHive format. The
dataset contains one CSV file per mircoarchitecture.

To process BHive throughput data for the Intel Skylake microarchitecture:

1.  Download the corresponding BHive CSV:

```bash
curl -L https://raw.githubusercontent.com/ithemal/bhive/5f1d50077ac0779fd227b261dcf517862c7104bd/benchmark/throughput/skl.csv > skl.csv
```

1.  Convert the downloaded CSV file into a `.tfrecord` file:

```bash
bazel run //gematria/datasets/python:import_from_bhive -- \
    --gematria_input_csv=skl.csv \
    --gematria_output_tfrecord=skl.tfrecord \
    --gematria_throughput_source_name="bhive: skl"
```

This will create a `skl.tfrecord` file containing imported basic blocks that can
be used for further downstream analysis. The `--gematria_input_csv` and
`--gematria_output_tfrecord` flags specify the path to the input CSV and the
path to the output `.tfrecord` files respectively. The
`--gematria_throughput_source_name` flag should be used to specify the method of
data collection. The protobuf data format that Gematria uses allows for storing
throughput information for multiple different microarchitectures for the same
basic block within a single protobuf. This allows for having a single dataset
with throughput data for all known microarchitectures that handles missing data
from specific sources gracefully. Typically labels like `bhive: skl`, `bhive:
hsw`, and `bhive: ivb` will be used in the same dataset (one per
microarchitecture) and then a model will be trained with multi-task training.

## Other CSV-based formats

There are other projects that also contain basic blocks and their associated
throughput data such Andreas Abel's
[uica-eval](https://github.com/andreas-abel/uiCA-eval). However, these projects
might use slightly different CSV formats and additional preprocessing may be
needed to import them.

## Parsing training data from ELF executables

TODO(boomanaiden154): Write documentation on this section once tooling for
extracting BBs from ELF binaries lands.
