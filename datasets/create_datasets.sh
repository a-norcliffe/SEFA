echo "Creating synthetic dataset"
python -m datasets.create_synthetic

echo "Creating cube dataset"
python -m datasets.create_cube

echo "Creating bank dataset"
python -m datasets.create_bank

echo "Creating california housing dataset"
python -m datasets.create_california_housing

echo "Creating miniboone dataset"
python -m datasets.create_miniboone

echo "Creating mnist dataset"
python -m datasets.create_mnist

echo "Creating fashion mnist dataset"
python -m datasets.create_fashion_mnist

echo "Creating metabric dataset"
python -m datasets.create_metabric

echo "Creating tcga dataset"
python -m datasets.create_tcga