from torchvision import datasets, transforms

# 1. 데이터 다운로드 및 로컬 저장
def download_fashion_mnist(root="fashion_mnist"):
    """
    Fashion MNIST 데이터셋을 다운로드합니다.
    :param root: 데이터를 저장할 디렉토리 경로
    """
    # Transform 설정 (이미지 정규화)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 학습용 데이터 다운로드
    train_dataset = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)

    # 테스트용 데이터 다운로드
    test_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    print(f"Fashion MNIST 데이터셋이 '{root}' 디렉토리에 저장되었습니다.")
    return train_dataset, test_dataset

# 실행
if __name__ == "__main__":
    train_dataset, test_dataset = download_fashion_mnist()