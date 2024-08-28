# Azure AI Studio에서 사용자 정의 Phi-3 모델을 파인 튜닝하고 Prompt flow와 통합하기

이 엔드투엔드(E2E) 샘플은 Microsoft Tech Community의 "[Azure AI Studio에서 사용자 정의 Phi-3 모델을 파인 튜닝하고 Prompt Flow와 통합하기](https://techcommunity.microsoft.com/t5/educator-developer-blog/fine-tune-and-integrate-custom-phi-3-models-with-prompt-flow-in/ba-p/4191726?wt.mc_id=studentamb_279723)" 가이드를 기반으로 합니다.

## Overview

이 E2E 샘플에서는 Phi-3 모델을 파인튜닝 하고 Azure AI Studio의 프롬프트 흐름과 통합하는 방법을 알아봅니다. Azure AI/ML Studio를 활용하여 사용자 지정 AI 모델을 배포하고 활용하기 위한 워크플로를 설정할 수 있습니다. 이 E2E 샘플은 세 가지 시나리오로 나뉩니다:

**시나리오 1: Azure 리소스 설정 및 미세 조정 준비**

**시나리오 2: Phi-3 모델 미세 조정 및 Azure 머신 러닝 스튜디오에 배포**

**시나리오 3: Prompt flow와 통합 및 Azure AI Studio에서 사용자 정의 모델과 채팅**

다음은 이 E2E 샘플의 개요입니다.

![Phi-3-FineTuning_PromptFlow_Integration Overview.](../imgs/FineTuning-PromptFlow-AIStudio/00-01-architecture.png)

### 목차

1. **[시나리오 1: Azure 리소스 설정 및 파인 튜닝 준비](#시나리오-1-azure-리소스-설정-및-파인-튜닝-준비)**
    - [Azure Machine Learning workspace 만들기](#azure-machine-learning-workspace-만들기)
    - [Azure 구독에서 GPU 할당량 요청](#azure-구독에서-gpu-할당량-요청)
    - [역할 할당 추가](#역할-할당-추가)
    - [프로젝트 설정](#프로젝트-설정)
    - [파인 튜닝을 위한 데이터셋 준비](#파인-튜닝을-위한-데이터셋-준비)

1. **[시나리오 2: Phi-3 모델 파인 튜닝 및 Azure Machine Learning Studio에 배포](#시나리오-2-phi-3-모델-파인-튜닝-및-azure-machine-learning-studio에-배포)**
    - [Phi-3 모델 파인 튜닝](#phi-3-모델-파인-튜닝)
    - [파인 튜닝된 Phi-3 모델 배포](#파인-튜닝된-phi-3-모델-배포)

1. **[시나리오 3: Prompt flow와 통합 및 Azure AI Studio에서 사용자 정의 모델과 채팅](#시나리오-3-prompt-flow와-통합-및-azure-ai-studio에서-사용자-정의-모델과-채팅)**
    - [사용자 정의 Phi-3 모델을 Prompt flow와 통합](#사용자-정의-phi-3-모델을-prompt-flow와-통합)
    - [사용자 정의 Phi-3 모델과 채팅](#사용자-정의-phi-3-모델과-채팅)

## 시나리오 1: Azure 리소스 설정 및 파인 튜닝 준비

### Azure Machine Learning workspace 만들기

1. 포털 페이지 상단의 **검색창**에 *azure machine learning*을 입력하고, 나타나는 옵션에서 **Azure Machine Learning**을 선택합니다.

    ![azure machine learning을 입력합니다.](../imgs/FineTuning-PromptFlow-AIStudio/01-01-type-azml.png)

2. 탐색 메뉴에서 **+ 만들기(+ Create)** 를 선택합니다.

3. 탐색 메뉴에서 **새 작업 영역(New workspace)** 을 선택합니다.

    ![새 작업 영역을 선택합니다.](../imgs/FineTuning-PromptFlow-AIStudio/01-02-select-new-workspace.png)

4. 다음 작업을 수행합니다:

    - Azure **구독(Subscription)** 을 선택합니다.
    - 사용할 **리소스 그룹(Resource group)** 을 선택합니다(필요시 새로 만듭니다).
    - **작업 영역 이름(Workspace Name)** 을 입력합니다. 고유한 값이어야 합니다.
    - 사용할 **지역(Region)** 을 선택합니다.
    - 사용할 **저장소 계정(Storage account)** 을 선택합니다(필요시 새로 만듭니다).
    - 사용할 **키 자격 증명 모음(Key vault)** 을 선택합니다(필요시 새로 만듭니다).
    - 사용할 **애플리케이션 인사이트(Application insights)** 를 선택합니다(필요시 새로 만듭니다).
    - 사용할 **컨테이너 레지스트리(Container registry)** 를 선택합니다(필요시 새로 만듭니다).

    ![azure machine learning을 채웁니다.](../imgs/FineTuning-PromptFlow-AIStudio/01-03-fill-AZML.png)

5. **검토 + 만들기(Review + Create)** 를 선택합니다.

6. **만들기(Create)** 를 선택합니다.

### Azure 구독에서 GPU 할당량 요청

이 튜토리얼에서는 GPU를 사용하여 Phi-3 모델을 파인 튜닝하고 배포하는 방법을 학습합니다. 파인 튜닝을 위해 *Standard_NC24ads_A100_v4* GPU를 사용할 것이며, 이는 할당량 요청이 필요합니다. 배포를 위해 *Standard_NC6s_v3* GPU를 사용할 것이며, 이 또한 할당량 요청이 필요합니다.

> [!NOTE]
>
> GPU 할당은 Pay-As-You-Go 구독(표준 구독 유형)만 사용할 수 있으며, 혜택 구독은 현재 지원되지 않습니다.
>

1. [Azure ML Studio](https://ml.azure.com/home?wt.mc_id=studentamb_279723)를 방문합니다.

1. *Standard NCADSA100v4 Family* 할당량을 요청하려면 다음 작업을 수행합니다:

    - 왼쪽 탭에서 **할당량(Quota)** 을 선택합니다.
    - 사용할 **가상 머신 계열**을 선택합니다. 예를 들어, *Standard_NC24ads_A100_v4* GPU가 포함된 **Standard NCADSA100v4 Family Cluster Dedicated vCPUs** 를 선택합니다.
    - 탐색 메뉴에서 **할당량 요청(Request quota)** 을 선택합니다.

        ![할당량 요청.](../imgs/FineTuning-PromptFlow-AIStudio/02-02-request-quota.png)

    - 할당량 요청 페이지에서 사용하려는 **새 코어 제한(New cores limit)** 을 입력합니다. 예를 들어, 24를 입력합니다.
    - 할당량 요청 페이지에서 **제출(Submit)** 을 선택하여 GPU 할당량을 요청합니다.

1. *Standard NCSv3 Family* 할당량을 요청하려면 다음 작업을 수행합니다:

    - 왼쪽 탭에서 **할당량(Quota)** 을 선택합니다.
    - 사용할 **가상 머신 계열**을 선택합니다. 예를 들어, *Standard_NC6s_v3* GPU가 포함된 **Standard NCSv3 Family Cluster Dedicated vCPUs**를 선택합니다.
    - 탐색 메뉴에서 **할당량 요청(Request quota)** 을 선택합니다.
    - 할당량 요청 페이지에서 사용하려는 **새 코어 제한(New cores limit)** 을 입력합니다. 예를 들어, 24를 입력합니다.
    - 할당량 요청 페이지에서 **제출(Submit)** 을 선택하여 GPU 할당량을 요청합니다.

### 역할 할당 추가

모델을 파인 튜닝하고 배포하려면 먼저 사용자 할당 관리 ID(User Assigned Managed Identity, UAI)를 생성하고 적절한 권한을 부여해야 합니다. 이 UAI는 배포 중 인증에 사용됩니다.

#### 사용자 할당 관리 ID(UAI) 생성

1. 포털 페이지 상단의 **검색창**에 *managed identities*를 입력하고, 나타나는 옵션에서 **Managed Identities**를 선택합니다.

    ![managed identities를 입력합니다.](../imgs/FineTuning-PromptFlow-AIStudio/03-01-type-managed-identities.png)

1. **+ 만들기(+ Create)** 를 선택합니다.

    ![만들기를 선택합니다.](../imgs/FineTuning-PromptFlow-AIStudio/03-02-select-create.png)

1. 다음 작업을 수행합니다:

    - Azure **구독(Subscription)** 을 선택합니다.
    - 사용할 **리소스 그룹(Resource group)** 을 선택합니다(필요시 새로 만듭니다).
    - 사용할 **지역(Region)** 을 선택합니다.
    - **이름(Name)** 을 입력합니다. 고유한 값이어야 합니다.

    ![정보를 입력합니다.](../imgs/FineTuning-PromptFlow-AIStudio/03-03-fill-managed-identities-1.png)

1. **검토 + 만들기(Review + create)** 를 선택합니다.

1. **+ 만들기(+ Create)** 를 선택합니다.

#### 관리 ID에 기여자 역할 할당 추가

1. 생성한 관리 ID 리소스로 이동합니다.

1. 왼쪽 탭에서 **Azure 역할 할당(Azure role assignments)** 을 선택합니다.

1. 탐색 메뉴에서 **+ 역할 할당 추가(+ Add role assignment)** 를 선택합니다.

1. 역할 할당 추가 페이지에서 다음 작업을 수행합니다:
    - **범위(Scope)** 를 **리소스 그룹(Resource group)** 으로 선택합니다.
    - Azure **구독(Subscription)** 을 선택합니다.
    - 사용할 **리소스 그룹(Resource group)** 을 선택합니다.
    - **역할(Role)** 을 **기여자(Contributor)** 로 선택합니다.

    ![기여자 역할을 입력합니다.](../imgs/FineTuning-PromptFlow-AIStudio/03-04-fill-contributor-role.png)

1. **저장(Save)** 을 선택합니다.

#### 관리 ID에 Storage Blob Data Reader 역할 할당 추가

1. 포털 페이지 상단의 **검색창**에 *storage accounts*를 입력하고, 나타나는 옵션에서 **Storage accounts**를 선택합니다.

    ![storage accounts를 입력합니다.](../imgs/FineTuning-PromptFlow-AIStudio/03-05-type-storage-accounts.png)

1. 생성한 Azure Machine Learning 작업 영역과 연관된 스토리지 계정을 선택합니다. 예를 들어, *finetunephistorage*.

1. 역할 할당 추가 페이지로 이동하려면 다음 작업을 수행합니다:

    - 생성한 Azure 스토리지 계정으로 이동합니다.
    - 왼쪽 탭에서 **액세스 제어(IAM) (Access Control (IAM))** 를 선택합니다.
    - 탐색 메뉴에서 **+ 추가(+ Add)** 를 선택합니다.
    - 탐색 메뉴에서 **역할 할당 추가(Add role assignment)** 를 선택합니다.

    ![역할 추가.](../imgs/FineTuning-PromptFlow-AIStudio/03-06-add-role.png)

1. 역할 할당 추가 페이지에서 다음 작업을 수행합니다:

    - 역할 페이지에서 **검색창**에 *Storage Blob Data Reader*를 입력하고, 나타나는 옵션에서 **Storage Blob Data Reader**를 선택합니다.
    - 역할 페이지에서 **다음(Next)** 을 선택합니다.
    - 구성원 페이지에서 **액세스 할당 대상(Assign access to)** 을 **Managed identity**로 선택합니다.
    - 구성원 페이지에서 **+ 구성원 선택(+ Select members)** 을 선택합니다.
    - 관리 ID 선택 페이지에서 Azure **구독(Subscription)** 을 선택합니다.
    - 관리 ID 선택 페이지에서 **관리 ID(Managed identity)** 를 선택합니다.
    - 관리 ID 선택 페이지에서 생성한 관리 ID를 선택합니다. 예: *finetunephi-managedidentity*.
    - 관리 ID 선택 페이지에서 **선택(Select)** 을 선택합니다.

    ![관리 ID 선택.](../imgs/FineTuning-PromptFlow-AIStudio/03-08-select-managed-identity.png)

1. **검토 + 할당(Review + assign)** 을 선택합니다.

#### 관리 ID에 AcrPull 역할 할당 추가

1. 포털 페이지 상단의 **검색창**에 *container registries*를 입력하고, 나타나는 옵션에서 **Container registries**를 선택합니다.

    ![container registries를 입력합니다.](../imgs/FineTuning-PromptFlow-AIStudio/03-09-type-container-registries.png)

1. Azure Machine Learning 작업 영역과 연관된 컨테이너 레지스트리를 선택합니다. 예를 들어, *finetunephicontainerregistry*.

1. 역할 할당 추가 페이지로 이동하려면 다음 작업을 수행합니다:

    - 왼쪽 탭에서 **액세스 제어(IAM) (Access Control (IAM))** 를 선택합니다.
    - 탐색 메뉴에서 **+ 추가(+ Add)** 를 선택합니다.
    - 탐색 메뉴에서 **역할 할당 추가(Add role assignment)** 를 선택합니다.

1. 역할 할당 추가 페이지에서 다음 작업을 수행합니다:

    - 역할 페이지에서 **검색창**에 *AcrPull*을 입력하고, 나타나는 옵션에서 **AcrPull**을 선택합니다.
    - 역할 페이지에서 **다음(Next)** 을 선택합니다.
    - 구성원 페이지에서 **액세스 할당 대상(Assign access to)** 을 **Managed identity**로 선택합니다.
    - 구성원 페이지에서 **+ 구성원 선택(+ Select members)** 을 선택합니다.
    - 관리 ID 선택 페이지에서 Azure **구독(Subscription)** 을 선택합니다.
    - 관리 ID 선택 페이지에서 **관리 ID(Managed identity)** 를 선택합니다.
    - 관리 ID 선택 페이지에서 생성한 관리 ID를 선택합니다. 예: *finetunephi-managedidentity*.
    - 관리 ID 선택 페이지에서 **선택(Select)** 을 선택합니다.
    - **검토 + 할당(Review + assign)** 을 선택합니다.

### 프로젝트 설정

파인 튜닝에 필요한 데이터셋을 다운로드하려면 로컬 환경을 설정해야 합니다.

이 실습에서는 다음을 수행합니다:

- 작업할 폴더를 생성합니다.
- 가상 환경을 생성합니다.
- 필요한 패키지를 설치합니다.
- 데이터셋을 다운로드하기 위한 *download_dataset.py* 파일을 생성합니다.

#### 작업할 폴더 생성

1. 터미널 창을 열고 다음 명령어를 입력하여 기본 경로에 *finetune-phi*라는 이름의 폴더를 생성합니다.

    ```console
    mkdir finetune-phi
    ```

2. 터미널에서 다음 명령어를 입력하여 생성한 *finetune-phi* 폴더로 이동합니다.

    ```console
    cd finetune-phi
    ```

#### 가상 환경 생성

1. 터미널에서 다음 명령어를 입력하여 *.venv*라는 이름의 가상 환경을 생성합니다.

    ```console
    python -m venv .venv
    ```

2. 터미널에서 다음 명령어를 입력하여 가상 환경을 활성화합니다.

    ```console
    .venv\Scripts\activate.bat
    ```

> [!NOTE]
> 가상 환경이 제대로 활성화되었다면 명령어 프롬프트 앞에 *(.venv)* 가 표시됩니다.

#### 필요한 패키지 설치

1. 터미널에서 다음 명령어를 입력하여 필요한 패키지를 설치합니다.

    ```console
    pip install datasets==2.19.1
    ```

#### `download_dataset.py` 파일 생성

> [!NOTE]
> 폴더 구조 예시:
>
> ```text
> └── YourUserName
> .    └── finetune-phi
> .        └── download_dataset.py
> ```

1. **Visual Studio Code**를 엽니다.

1. 메뉴 바에서 **파일(File)** 을 선택합니다.

1. **폴더 열기(Open Folder)** 를 선택합니다.

1. *C:\Users\yourUserName\finetune-phi* 경로에 있는, 생성한 *finetune-phi* 폴더를 선택합니다.

    ![생성한 폴더를 선택합니다.](../imgs/FineTuning-PromptFlow-AIStudio/04-01-open-project-folder.png)

1. Visual Studio Code의 왼쪽 창에서 마우스 오른쪽 버튼을 클릭하고 **새 파일(New File)** 을 선택하여 *download_dataset.py*라는 새 파일을 생성합니다.

    ![새 파일을 생성합니다.](../imgs/FineTuning-PromptFlow-AIStudio/04-02-create-new-file.png)

### 파인 튜닝을 위한 데이터셋 준비

이 실습에서는 *download_dataset.py* 파일을 실행하여 *ultrachat_200k* 데이터셋을 로컬 환경에 다운로드합니다. 그런 다음, 이 데이터셋을 사용하여 Azure Machine Learning에서 Phi-3 모델을 파인 튜닝할 것입니다.

이 실습에서는 다음을 수행합니다:

- 데이터셋을 다운로드하기 위해 *download_dataset.py* 파일에 코드를 추가합니다.
- *download_dataset.py* 파일을 실행하여 로컬 환경에 데이터셋을 다운로드합니다.

#### *download_dataset.py*를 사용하여 데이터셋 다운로드

1. Visual Studio Code에서 *download_dataset.py* 파일을 엽니다.

1. *download_dataset.py* 파일에 다음 코드를 추가합니다.

    ```python
    import json
    import os
    from datasets import load_dataset

    def load_and_split_dataset(dataset_name, config_name, split_ratio):
        """
        Load and split a dataset.
        """
        # Load the dataset with the specified name, configuration, and split ratio
        dataset = load_dataset(dataset_name, config_name, split=split_ratio)
        print(f"Original dataset size: {len(dataset)}")
        
        # Split the dataset into train and test sets (80% train, 20% test)
        split_dataset = dataset.train_test_split(test_size=0.2)
        print(f"Train dataset size: {len(split_dataset['train'])}")
        print(f"Test dataset size: {len(split_dataset['test'])}")
        
        return split_dataset

    def save_dataset_to_jsonl(dataset, filepath):
        """
        Save a dataset to a JSONL file.
        """
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Open the file in write mode
        with open(filepath, 'w', encoding='utf-8') as f:
            # Iterate over each record in the dataset
            for record in dataset:
                # Dump the record as a JSON object and write it to the file
                json.dump(record, f)
                # Write a newline character to separate records
                f.write('\n')
        
        print(f"Dataset saved to {filepath}")

    def main():
        """
        Main function to load, split, and save the dataset.
        """
        # Load and split the ULTRACHAT_200k dataset with a specific configuration and split ratio
        dataset = load_and_split_dataset("HuggingFaceH4/ultrachat_200k", 'default', 'train_sft[:1%]')
        
        # Extract the train and test datasets from the split
        train_dataset = dataset['train']
        test_dataset = dataset['test']

        # Save the train dataset to a JSONL file
        save_dataset_to_jsonl(train_dataset, "data/train_data.jsonl")
        
        # Save the test dataset to a separate JSONL file
        save_dataset_to_jsonl(test_dataset, "data/test_data.jsonl")

    if __name__ == "__main__":
        main()

    ```

1. 터미널에서 다음 명령어를 입력하여 스크립트를 실행하고 로컬 환경에 데이터셋을 다운로드합니다.

    ```console
    python download_dataset.py
    ```

1. 데이터셋이 로컬 *finetune-phi/data* 디렉터리에 성공적으로 저장되었는지 확인합니다.

> [!NOTE]
>
> #### 데이터셋 크기 및 파인 튜닝 시간에 대한 참고 사항
>
> 이 튜토리얼에서는 데이터셋의 1%만 사용합니다 (`split='train[:1%]'`). 이는 데이터의 양을 크게 줄여 업로드 및 파인 튜닝 프로세스 속도를 높입니다. 학습 시간과 모델 성능 간의 균형을 맞추기 위해 이 비율을 조정할 수 있습니다. 데이터셋의 작은 부분을 사용하는 것은 파인 튜닝에 필요한 시간을 줄여 튜토리얼 과정을 보다 관리하기 쉽게 만듭니다.

## 시나리오 2: Phi-3 모델 파인 튜닝 및 Azure Machine Learning Studio에 배포

### Phi-3 모델 파인 튜닝

이 실습에서는 Azure Machine Learning Studio에서 Phi-3 모델을 파인 튜닝합니다.

이 실습에서는 다음을 수행합니다:

- 파인 튜닝을 위한 컴퓨터 클러스터를 생성합니다.
- Azure Machine Learning Studio에서 Phi-3 모델을 파인 튜닝합니다.

#### 파인 튜닝을 위한 컴퓨터 클러스터 생성

1. [Azure ML Studio](https://ml.azure.com/home?wt.mc_id=studentamb_279723)를 방문합니다.

1. 왼쪽 탭에서 **Compute**를 선택합니다.

1. 탐색 메뉴에서 **Compute clusters**를 선택합니다.

1. **+ New**를 선택합니다.

    ![Compute 선택.](../imgs/FineTuning-PromptFlow-AIStudio/06-01-select-compute.png)

1. 다음 작업을 수행합니다:

    - 사용할 **지역(Region)** 을 선택합니다.
    - **가상 머신 등급(Virtual machine tier)** 을 **Dedicated**로 선택합니다.
    - **가상 머신 유형(Virtual machine type)** 을 **GPU**로 선택합니다.
    - **가상 머신 크기 필터(Virtual machine size filter)** 를 **모든 옵션에서 선택(Select from all options)** 으로 설정합니다.
    - **가상 머신 크기(Virtual machine size)** 를 **Standard_NC24ads_A100_v4**로 선택합니다.

    ![클러스터 생성.](../imgs/FineTuning-PromptFlow-AIStudio/06-02-create-cluster.png)

1. **Next**를 선택합니다.

1. 다음 작업을 수행합니다:

    - **Compute 이름(Compute name)** 을 입력합니다. 고유한 값이어야 합니다.
    - **최소 노드 수(Minimum number of nodes)** 를 **0**으로 설정합니다.
    - **최대 노드 수(Maximum number of nodes)** 를 **1**로 설정합니다.
    - **유휴 상태 후 축소 시간(Idle seconds before scale down)** 을 **120**으로 설정합니다.

    ![클러스터 생성.](../imgs/FineTuning-PromptFlow-AIStudio/06-03-create-cluster.png)

1. **Create**를 선택합니다.

#### Phi-3 모델 파인 튜닝

1. [Azure ML Studio](https://ml.azure.com/home?wt.mc_id=studentamb_279723)를 방문합니다.

1. 생성한 Azure Machine Learning 작업 영역을 선택합니다.

    ![생성한 작업 영역 선택.](../imgs/FineTuning-PromptFlow-AIStudio/06-04-select-workspace.png)

1. 다음 작업을 수행합니다:

    - 왼쪽 탭에서 **Model catalog**를 선택합니다.
    - **검색창**에 *phi-3-mini-4k*를 입력하고, 나타나는 옵션에서 **Phi-3-mini-4k-instruct**를 선택합니다.

    ![phi-3-mini-4k 입력.](../imgs/FineTuning-PromptFlow-AIStudio/06-05-type-phi-3-mini-4k.png)

1. 탐색 메뉴에서 **Fine-tune**을 선택합니다.

    ![파인 튜닝 선택.](../imgs/FineTuning-PromptFlow-AIStudio/06-06-select-fine-tune.png)

1. 다음 작업을 수행합니다:

    - **Select task type**을 **Chat completion**으로 선택합니다.
    - **+ Select data**를 선택하여 **훈련 데이터(Training data)**를 업로드합니다.
    - 검증 데이터 업로드 유형을 **다른 검증 데이터 제공(Provide different validation data)** 으로 선택합니다.
    - **+ Select data**를 선택하여 **검증 데이터(Validation data)** 를 업로드합니다.

    ![파인 튜닝 페이지 입력.](../imgs/FineTuning-PromptFlow-AIStudio/06-07-fill-finetuning.png)

    > [!TIP]
    >
    > **고급 설정(Advanced settings)**을 선택하여 **learning_rate** 및 **lr_scheduler_type**과 같은 설정을 사용자 지정하여 파인 튜닝 프로세스를 최적화할 수 있습니다.

1. **Finish**를 선택합니다.

1. 이 실습에서는 Azure Machine Learning을 사용하여 Phi-3 모델을 성공적으로 파인 튜닝했습니다. 파인 튜닝 과정은 상당한 시간이 걸릴 수 있습니다. 파인 튜닝 작업을 실행한 후 완료될 때까지 기다려야 합니다. Azure Machine Learning 작업 영역의 왼쪽 탭에서 Jobs 탭으로 이동하여 파인 튜닝 작업의 상태를 모니터링할 수 있습니다. 다음 시리즈에서는 파인 튜닝된 모델을 배포하고 Prompt flow와 통합할 것입니다.

    ![파인 튜닝 작업 보기.](../imgs/FineTuning-PromptFlow-AIStudio/06-08-output.png)

### 파인 튜닝된 Phi-3 모델 배포

파인 튜닝된 Phi-3 모델을 Prompt flow와 통합하려면 실시간 추론이 가능하도록 모델을 배포해야 합니다. 이 과정에는 모델 등록, 온라인 엔드포인트 생성 및 모델 배포가 포함됩니다.

이 실습에서는 다음을 수행합니다:

- Azure Machine Learning 작업 영역에 파인 튜닝된 모델을 등록합니다.
- 온라인 엔드포인트를 생성합니다.
- 등록된 파인 튜닝된 Phi-3 모델을 배포합니다.

#### 파인 튜닝된 모델 등록

1. [Azure ML Studio](https://ml.azure.com/home?wt.mc_id=studentamb_279723)를 방문합니다.

1. 생성한 Azure Machine Learning 작업 영역을 선택합니다.

    ![생성한 작업 영역 선택.](../imgs/FineTuning-PromptFlow-AIStudio/06-04-select-workspace.png)

1. 왼쪽 탭에서 **Models**를 선택합니다.
1. **+ Register**를 선택합니다.
1. **From a job output**을 선택합니다.

    ![모델 등록.](../imgs/FineTuning-PromptFlow-AIStudio/07-01-register-model.png)

1. 생성한 작업(job)을 선택합니다.

    ![작업 선택.](../imgs/FineTuning-PromptFlow-AIStudio/07-02-select-job.png)

1. **Next**를 선택합니다.

1. **Model type**을 **MLflow**로 설정합니다.

1. **Job output**이 선택되었는지 확인합니다. 자동으로 선택되어 있어야 합니다.

    ![출력 선택.](../imgs/FineTuning-PromptFlow-AIStudio/07-03-select-output.png)

1. **Next**를 선택합니다.

1. **Register**를 선택합니다.

    ![등록 선택.](../imgs/FineTuning-PromptFlow-AIStudio/07-04-register.png)

1. 왼쪽 탭의 **Models** 메뉴로 이동하여 등록된 모델을 확인할 수 있습니다.

    ![등록된 모델.](../imgs/FineTuning-PromptFlow-AIStudio/07-05-registered-model.png)

#### 파인 튜닝된 모델 배포

1. 생성한 Azure Machine Learning 작업 영역으로 이동합니다.

1. 왼쪽 탭에서 **Endpoints**를 선택합니다.

1. 탐색 메뉴에서 **Real-time endpoints**를 선택합니다.

    ![엔드포인트 생성.](../imgs/FineTuning-PromptFlow-AIStudio/07-06-create-endpoint.png)

1. **Create**를 선택합니다.

1. 생성한 등록된 모델을 선택합니다.

    ![등록된 모델 선택.](../imgs/FineTuning-PromptFlow-AIStudio/07-07-select-registered-model.png)

1. **Select**를 선택합니다.

1. 다음 작업을 수행합니다:

    - **Virtual machine**을 *Standard_NC6s_v3*로 선택합니다.
    - 사용할 **인스턴스 수(Instance count)** 를 선택합니다. 예를 들어, *1*.
    - **Endpoint**를 **New**로 설정하여 새 엔드포인트를 생성합니다.
    - **엔드포인트 이름(Endpoint name)** 을 입력합니다. 고유한 값이어야 합니다.
    - **배포 이름(Deployment name)** 을 입력합니다. 고유한 값이어야 합니다.

    ![배포 설정 입력.](../imgs/FineTuning-PromptFlow-AIStudio/07-08-deployment-setting.png)

1. **Deploy**를 선택합니다.

> [!WARNING]
> 추가 비용이 발생하지 않도록 Azure Machine Learning 작업 영역에서 생성한 엔드포인트를 삭제하는 것을 잊지 마십시오.
>

#### Azure Machine Learning 작업 영역에서 배포 상태 확인

1. 생성한 Azure Machine Learning 작업 영역으로 이동합니다.

1. 왼쪽 탭에서 **Endpoints**를 선택합니다.

1. 생성한 엔드포인트를 선택합니다.

    ![엔드포인트 선택](../imgs/FineTuning-PromptFlow-AIStudio/07-09-check-deployment.png)

1. 이 페이지에서 배포 과정 동안 엔드포인트를 관리할 수 있습니다.

> [!NOTE]
> 배포가 완료되면 **Live traffic**이 **100%**로 설정되어 있는지 확인하십시오. 설정되어 있지 않다면 **Update traffic**을 선택하여 트래픽 설정을 조정하십시오. 트래픽이 0%로 설정되어 있으면 모델을 테스트할 수 없습니다.
>
> ![트래픽 설정.](../imgs/FineTuning-PromptFlow-AIStudio/07-10-set-traffic.png)
>

## 시나리오 3: Prompt flow와 통합 및 Azure AI Studio에서 사용자 정의 모델과 채팅

### 사용자 정의 Phi-3 모델을 Prompt flow와 통합

파인 튜닝된 모델을 성공적으로 배포한 후, 이제 Prompt Flow와 통합하여 실시간 애플리케이션에서 모델을 사용할 수 있습니다. 이를 통해 사용자 정의 Phi-3 모델을 사용한 다양한 상호작용 작업이 가능합니다.

이 실습에서는 다음을 수행합니다:

- Azure AI Studio Hub 생성.
- Azure AI Studio 프로젝트 생성.
- Prompt flow 생성.
- 파인 튜닝된 Phi-3 모델을 위한 사용자 정의 연결 추가.
- Prompt flow를 설정하여 사용자 정의 Phi-3 모델과 채팅.

> [!NOTE]
> Azure ML Studio를 사용하여 Promptflow와 통합할 수도 있습니다. 동일한 통합 프로세스를 Azure ML Studio에 적용할 수 있습니다.

#### Azure AI Studio Hub 생성

프로젝트를 생성하기 전에 Hub를 생성해야 합니다. Hub는 리소스 그룹과 같은 역할을 하며, Azure AI Studio 내에서 여러 프로젝트를 구성하고 관리할 수 있습니다.

1. [Azure AI Studio](https://ai.azure.com/?wt.mc_id=studentamb_279723)를 방문합니다.

1. 왼쪽 탭에서 **All hubs**를 선택합니다.

1. 탐색 메뉴에서 **+ New hub**를 선택합니다.

    ![Hub 생성.](../imgs/FineTuning-PromptFlow-AIStudio/08-01-create-hub.png)

1. 다음 작업을 수행합니다:

    - **Hub 이름(Hub name)** 을 입력합니다. 고유한 값이어야 합니다.
    - Azure **구독(Subscription)** 을 선택합니다.
    - 사용할 **리소스 그룹(Resource group)** 을 선택합니다(필요시 새로 만듭니다).
    - 사용할 **위치(Location)**를 선택합니다.
    - 사용할 **Azure AI 서비스(Azure AI Services)** 를 연결합니다(필요시 새로 만듭니다).
    - **Azure AI 검색(Azure AI Search)** 을 **연결 건너뛰기(Skip connecting)**로 설정합니다.

    ![Hub 정보 입력.](../imgs/FineTuning-PromptFlow-AIStudio/08-02-fill-hub.png)

1. **Next**를 선택합니다.

#### Azure AI Studio 프로젝트 생성

1. 생성한 Hub에서 왼쪽 탭에서 **All projects**를 선택합니다.

1. 탐색 메뉴에서 **+ New project**를 선택합니다.

    ![새 프로젝트 선택.](../imgs/FineTuning-PromptFlow-AIStudio/08-04-select-new-project.png)

1. **프로젝트 이름(Project name)** 을 입력합니다. 고유한 값이어야 합니다.

    ![프로젝트 생성.](../imgs/FineTuning-PromptFlow-AIStudio/08-05-create-project.png)

1. **프로젝트 생성(Create a project)** 을 선택합니다.

#### 파인 튜닝된 Phi-3 모델을 위한 사용자 정의 연결 추가

사용자 정의 Phi-3 모델을 Prompt flow와 통합하려면 모델의 엔드포인트와 키를 사용자 정의 연결에 저장해야 합니다. 이 설정을 통해 Prompt flow에서 사용자 정의 Phi-3 모델에 액세스할 수 있습니다.

#### 파인 튜닝된 Phi-3 모델의 API 키 및 엔드포인트 URI 설정

1. [Azure ML Studio](https://ml.azure.com/home?wt.mc_id=studentamb_279723)를 방문합니다.

1. 생성한 Azure Machine Learning 작업 영역으로 이동합니다.

1. 왼쪽 탭에서 **Endpoints**를 선택합니다.

    ![엔드포인트 선택.](../imgs/FineTuning-PromptFlow-AIStudio/08-06-select-endpoints.png)

1. 생성한 엔드포인트를 선택합니다.

    ![생성한 엔드포인트 선택.](../imgs/FineTuning-PromptFlow-AIStudio/08-07-select-endpoint-created.png)

1. 탐색 메뉴에서 **Consume**을 선택합니다.

1. **REST endpoint**와 **Primary key**를 복사합니다.

    ![API 키와 엔드포인트 URI 복사.](../imgs/FineTuning-PromptFlow-AIStudio/08-08-copy-endpoint-key.png)

#### 사용자 정의 연결 추가

1. [Azure AI Studio](https://ai.azure.com/?wt.mc_id=studentamb_279723)를 방문합니다.

1. 생성한 Azure AI Studio 프로젝트로 이동합니다.

1. 생성한 프로젝트에서 왼쪽 탭에서 **Settings**를 선택합니다.

1. **+ New connection**을 선택합니다.

    ![새 연결 선택.](../imgs/FineTuning-PromptFlow-AIStudio/08-09-select-new-connection.png)

1. 탐색 메뉴에서 **Custom keys**를 선택합니다.

    ![Custom keys 선택.](../imgs/FineTuning-PromptFlow-AIStudio/08-10-select-custom-keys.png)

1. 다음 작업을 수행합니다:

    - **+ Add key value pairs**를 선택합니다.
    - 키 이름에 **endpoint**를 입력하고, Azure ML Studio에서 복사한 엔드포인트를 값 필드에 붙여넣습니다.
    - **+ Add key value pairs**를 다시 선택합니다.
    - 키 이름에 **key**를 입력하고, Azure ML Studio에서 복사한 키를 값 필드에 붙여넣습니다.
    - 키를 추가한 후 **is secret**을 선택하여 키가 노출되지 않도록 설정합니다.

    ![연결 추가.](../imgs/FineTuning-PromptFlow-AIStudio/08-11-add-connection.png)

1. **Add connection**을 선택합니다.

#### Prompt flow 생성

Azure AI Studio에서 사용자 정의 연결을 추가했습니다. 이제 다음 단계를 따라 Prompt flow를 생성합니다. 그런 다음 이 Prompt flow를 사용자 정의 연결에 연결하여 Prompt flow 내에서 파인 튜닝된 모델을 사용할 수 있도록 합니다.

1. 생성한 Azure AI Studio 프로젝트로 이동합니다.

1. 왼쪽 탭에서 **Prompt flow**를 선택합니다.

1. 탐색 메뉴에서 **+ Create**를 선택합니다.

    ![Promptflow 선택.](../imgs/FineTuning-PromptFlow-AIStudio/08-12-select-promptflow.png)

1. 탐색 메뉴에서 **Chat flow**를 선택합니다.

    ![Chat flow 선택.](../imgs/FineTuning-PromptFlow-AIStudio/08-13-select-flow-type.png)

1. 사용할 **폴더 이름(Folder name)**을 입력합니다.

    ![이름 입력.](../imgs/FineTuning-PromptFlow-AIStudio/08-14-enter-name.png)

1. **Create**를 선택합니다.

#### 사용자 정의 Phi-3 모델과 채팅할 수 있도록 Prompt flow 설정

파인 튜닝된 Phi-3 모델을 Prompt flow에 통합해야 합니다. 그러나 기존 Prompt flow는 이 목적에 맞게 설계되지 않았으므로, 사용자 정의 모델의 통합을 가능하게 하려면 Prompt flow를 재설계해야 합니다.

1. Prompt flow에서 기존 플로우를 재구성하기 위해 다음 작업을 수행합니다:

    - **Raw file mode**를 선택합니다.
    - *flow.dag.yml* 파일에 있는 기존 코드를 모두 삭제합니다.
    - 아래 코드를 *flow.dag.yml* 파일에 추가합니다.

        ```yml
        inputs:
          input_data:
            type: string
            default: "Who founded Microsoft?"

        outputs:
          answer:
            type: string
            reference: ${integrate_with_promptflow.output}

        nodes:
        - name: integrate_with_promptflow
          type: python
          source:
            type: code
            path: integrate_with_promptflow.py
          inputs:
            input_data: ${inputs.input_data}
        ```

    - **Save**를 선택합니다.

    ![Raw file mode 선택.](../imgs/FineTuning-PromptFlow-AIStudio/08-15-select-raw-file-mode.png)

1. *integrate_with_promptflow.py* 파일에 사용자 정의 Phi-3 모델을 Prompt flow에서 사용할 수 있도록 아래 코드를 추가합니다.

    ```python
    import logging
    import requests
    from promptflow import tool
    from promptflow.connections import CustomConnection

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG
    )
    logger = logging.getLogger(__name__)

    def query_phi3_model(input_data: str, connection: CustomConnection) -> str:
        """
        Send a request to the Phi-3 model endpoint with the given input data using Custom Connection.
        """

        # "connection" is the name of the Custom Connection, "endpoint", "key" are the keys in the Custom Connection
        endpoint_url = connection.endpoint
        api_key = connection.key

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "input_data": {
                "input_string": [
                    {"role": "user", "content": input_data}
                ],
                "parameters": {
                    "temperature": 0.7,
                    "max_new_tokens": 128
                }
            }
        }
        try:
            response = requests.post(endpoint_url, json=data, headers=headers)
            response.raise_for_status()
            
            # Log the full JSON response
            logger.debug(f"Full JSON response: {response.json()}")

            result = response.json()["output"]
            logger.info("Successfully received response from Azure ML Endpoint.")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Azure ML Endpoint: {e}")
            raise

    @tool
    def my_python_tool(input_data: str, connection: CustomConnection) -> str:
        """
        Tool function to process input data and query the Phi-3 model.
        """
        return query_phi3_model(input_data, connection)

    ```

    ![prompt flow 를 추가합니다.](../imgs/FineTuning-PromptFlow-AIStudio/08-16-paste-promptflow-code.png)

> [!NOTE]
> Prompt flow를 Azure AI Studio에서 사용하는 방법에 대한 자세한 내용은 [Azure AI Studio의 Prompt flow](https://learn.microsoft.com/azure/ai-studio/how-to/prompt-flow)를 참조하십시오.

1. **Chat input**과 **Chat output**을 선택하여 모델과 채팅할 수 있도록 설정합니다.

    ![입력 및 출력 선택.](../imgs/FineTuning-PromptFlow-AIStudio/08-17-select-input-output.png)

1. 이제 사용자 정의 Phi-3 모델과 채팅할 준비가 되었습니다. 다음 실습에서는 Prompt flow를 시작하고, 이를 사용하여 파인 튜닝된 Phi-3 모델과 채팅하는 방법을 배웁니다.

> [!NOTE]
>
> 재구성된 플로우는 아래 이미지와 같이 보여야 합니다:
>
> ![플로우 예시.](../imgs/FineTuning-PromptFlow-AIStudio/08-18-graph-example.png)
>

### 사용자 정의 Phi-3 모델과 채팅

이제 파인 튜닝된 사용자 정의 Phi-3 모델을 Prompt flow와 통합했으므로, 모델과의 상호작용을 시작할 준비가 되었습니다. 이 실습에서는 Prompt flow를 설정하고 모델과의 채팅을 시작하는 과정을 안내합니다. 이 단계를 따라가면, 파인 튜닝된 Phi-3 모델의 기능을 다양한 작업과 대화에 완전히 활용할 수 있게 됩니다.

- Prompt flow를 사용하여 사용자 정의 Phi-3 모델과 채팅.

#### Prompt flow 시작

1. Prompt flow를 시작하려면 **Start compute sessions**을 선택합니다.

    ![계산 세션 시작.](../imgs/FineTuning-PromptFlow-AIStudio/09-01-start-compute-session.png)

1. **Validate and parse input**을 선택하여 매개변수를 갱신합니다.

    ![입력 유효성 검사.](../imgs/FineTuning-PromptFlow-AIStudio/09-02-validate-input.png)

1. **connection**의 **Value**를 생성한 사용자 정의 연결로 선택합니다. 예: *connection*.

    ![연결 선택.](../imgs/FineTuning-PromptFlow-AIStudio/09-03-select-connection.png)

#### 사용자 정의 모델과 채팅

1. **Chat**을 선택합니다.

    ![채팅 선택.](../imgs/FineTuning-PromptFlow-AIStudio/09-04-select-chat.png)

1. 다음은 결과 예시입니다: 이제 사용자 정의 Phi-3 모델과 채팅할 수 있습니다. 파인 튜닝에 사용된 데이터를 기반으로 질문하는 것이 좋습니다.

    ![Prompt flow와 채팅.](../imgs/FineTuning-PromptFlow-AIStudio/09-05-chat-with-promptflow.png)
