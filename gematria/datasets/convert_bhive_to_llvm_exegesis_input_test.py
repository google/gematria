from lit.main import main
import sys

# Lit expects the test folder path to be specifided on the command-line, which
# is usually passed in through CMake. Bazel doesn't support this configuration,
# so we manually add the path here.
sys.argv.append(
    "./gematria/datasets/convert_bhive_to_llvm_exegesis_input_tests"
)
sys.argv.append("-vv")

if __name__ == "__main__":
  main()
