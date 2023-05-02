import click

from karyohmm import EuploidyHMM, MetaHMM


@click.command()  # E: expected 2 blank lines, found 1
@click.option(
    "--input", "-i", required=True, type=str, help="Input data file for PGT Data."
)
@click.option(
    "--out", "-o", required=True, type=str, help="Output file."
)  # E: line too long (94 > 79 characters)
def main(input, out):
    """Main CLI entrypoint for calling karyohmm."""
    print(input, out)
    pass
