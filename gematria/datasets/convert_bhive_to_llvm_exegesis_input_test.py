from lit.main import main
import os
import sys

sys.argv.append("./gematria/datasets/tests")

with open('/tmp/test', 'w') as test_file:
    test_file.write(str(os.listdir('./gematria/datasets/tests')))

if __name__ == '__main__':
    main()
