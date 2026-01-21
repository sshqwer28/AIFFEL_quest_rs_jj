### 비선형적이란?
- 입력과 출력 사이의 관계까 직선(일차식)으로 표현되지 않는다는 뜻 -> 복잡한 패턴을 만들기 위함

<br>

### 초기 CNN이 Conv → BN → ReLU 순서로 이루어진 이유?
1. 먼저 특징을 만들어야 한다 → Conv (Weight)
2. 그런데 Conv 결과는 분포가 엉망이다
    - 평균이 계속 바뀌고
    - 분산이 층마다 달라지고
    - 학습이 불안정해진다 → 그래서 BN이 필요해졌다
3. 그런데 BN 뒤에는 비선형성이 있어야 표현력이 생긴다 → ReLU

<br>

### ResNet pre-activation에서의 순서가 초기의 CNN과 과정이 다른 이유는?
#### ```BN → ReLU → Weight```
1. 위와 같은 순서를 쓰는 이유는 "가중치가 가장 깨끗한 상태의 입력을 보게 하자"는 철학 때문
2. ResNet pre-activation의 핵심 목표
    1. Shortcut은 완전한 identity로 둔다
    2. Residual 경로에서는 계산이 가장 안정적인 상태에서 이루어지게 한다
3. **의문** → "왜 굳이 정리가 안 된 신호를 먼저 Conv에 넣는가? 정리부터 하고, 의미 있는 부분만 남긴 다음에 Conv를 통과시키면 되지 않은가?
4. 이러한 의문에서 출발하여 ```BN → ReLU → Conv```(2번 반복, 마지막에는 shortcut(x)와 더함) 순서로 개발됨
5. 즉, ```정리 → 선택 → 변형```
6. 뒤에는 더이상 ReLU를 붙이지 않음 → shortcut을 더한 뒤에 ReLU를 붙이면 shortcut 경로가 다시 잘리고 identity가 깨지기 때문임

<img width="347" height="402" alt="image" src="https://github.com/user-attachments/assets/8bdb7843-4ba8-463a-a331-47f66468a0b0" />

<br>

### DenseNet에서 Connection 개수 세기
1. 기본 식은 ```L(L + 1)/2```
2. connection 이라는 건, DenseNet에서는 각 층이 자기 뒤에 있는 모든 층과 직접 연결된다는 뜻임
3. 그래서 레이어가 6개 라면 connection이 연결되는 점은 7개(1 + 6)이 되는 것임

<br>

### Growth Rate?
1. Dense Block 내에서 레이어 하나당 추가되는 채널 수
2. Block이 끝나면 Transition Layer로 가서 이걸 다시 압축한다.

<br>

### bottleneck 레이어, transition 레이어, composite function
1. Composite Function
   - 한 레이어의 기본 연산 묶음
   - ```BN → ReLU → Conv``` : 이 세 개를 합쳐서 하나의 함수로 봄
   - 즉, ```정규화 → 비선형 → 합성곱```을 하나의 레이어 동작으로 묶은 것
2. Bottleneck Layer
   - Dense Block 내 각 레이어의 특정 형태의 내부 구조를 의미함
   - 구조 : ```[BN → ReLU → Conv(1×1)] → [BN → ReLU → Conv(3×3)]```
   - Conv(1X1)
       - 채널 수를 줄이는 압축 단계
       - 연산량 감소
   - Conv(3X3)
       - 실제 공간적 특징을 뽑는 단계
   - 좁은 통로 (1x1 Conv)를 한 번 통과시킨 후, 뒤의 큰 연산(3x3 Conv)을 가볍게 만든다는 의미임
3. Transition Layer
   - Dense Block과 Dense Block 사이에 들어가는 정리용 레이어
   - 구조는 보통: ```BN → Conv(1X1) → Avg Pooling```
   - 위 논문에서는 compression factor을 θ = 0.5으로 설정하였음

<br>

### NASNet
1. 사람이 Convolution Block을 설계하는 것이 아니라, 강화학습과 RNN을 활용하여 block을 자동으로 설계하는 모델
2. 다른 말로 하면, 알고리즘이 normal cell, reduction cell의 내부 구성과 구조를 자동으로 탐색하여, 최적의 CNN구조를 만들도록 함
3. 사람이 경험과 직관으로 네트워크를 설계하는 것이 아니라 알고리즘이 설계!
4. AutoML의 시작점이 된 모델
5. 인간 설계보다 더 효율적인 구조 발견이 가능하다는 장점이 있지만, 탐색 비용이 매우 큼. 그래서 실무에서는 직접 NAS를 돌리기 어렵고, 이미 공개된 구조를 사용하는 경우가 많음
6. Convolution Cell
   - Normal Cell
     - 표현을 더 정교하고 구체적으로 만드는 단계
     - 해상도(H, W)는 유지하면서 특징을 풍부하게 함
   - Reduction Cell
     - 중요한 정보를 압축하고 추상화하는 단계
     - 해상도를 줄이고, 채널 수를 늘려서 핵심 정보만 남김
7. 자동 탐색 대상, 사람이 고정하는 것
   1. 자동으로 탐색되는 것
      - Cell 내부 구조
      - 어떤 연산을 쓸지
        - 3X3 Conv
        - 5X5 Conv
        - Separable Conv
        - Max / Avg Pooling
        - Identity (Skip Connection)
      - 어떤 노드끼리 연결할지
      - 병렬 구조를 어떻게 만들지
      - 두 feature map을 어떻게 합질지
     
    2. 사람이 고정하는 것
      - Cell의 배치 순서
      - Reduction cell을 몇 번 넣을지
      - 전체 네트워크 깊이
      - 초기 채널 수
      - 채널 증가 비율
      - 입력 해상도
  
<br>

### EfficientNet
1. 자동으로 효율적으로 모델을 키우는 방법을 제시한 모델
1. Compound Scailing
   - Depth, Width, Resolution을 균형있게(비율 규칙에 따라) 동시에 키우는 것
   - Depth(α) : 레이어(네트워크)가 얼마나 깊게 쌓여 있는가
   - Width(β) : 각 레이어(채널)가 얼마나 두꺼운가
   - Resolution(γ) : feature map의 공간 크기(이미지 H X W 크기)
   - "ϕ"는 사람이 직접 모델의 크기를 몇 배로 확대할지 정하는 배율 버튼 같은 것
2. NAS와 다른 점
   - NAS는 Cell 구조를 찾는 모델
   - EfficientNet은 NAS로 찾은 하나의 좋은 네트워크(B0)를 기반으로 효율적으로 모델을 키우는 방법을 제시한 모델
   - NAS는 초기 1회만 사용
3. 전체 과정을 간단히 정리하면...
   1. NAS로 최적의 기본 네트워크 구조(B0)를 찾음
   2. B0 구조에서 Grid Search로 최적의 α, β, γ를 찾음
   3. 배율 ϕ를 정해서 모델 크기를 선택함
   4. 정해진 구조를 기반으로 모델을 새로 학습시킴



 





