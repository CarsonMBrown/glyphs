import argparse

# Base argument
arg_parser = argparse.ArgumentParser(prog="Kleio",
                                     description="A pipeline for optical character classification and localization of "
                                                 "glyphs on degraded papyri",
                                     add_help=True,
                                     epilog=""
                                     )

actions = arg_parser.add_mutually_exclusive_group()

# allow sub commands
sub_parsers = arg_parser.add_subparsers(dest="hello", required=True)

# add sub-parser for transcribe command
transcribe_parser = sub_parsers.add_parser("transcribe")
transcribe_parser.add_argument("verbose")
transcribe_parser.set_defaults(func=None)

if __name__ == '__main__':
    arg_parser.print_help()

# args = parser.parse_args()
# if args.func:
#     args.func(args)
