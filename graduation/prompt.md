목표표:
1) 예심 준비를 위해서는 일단 80% 정도 완성된 학위 논문 (양식에 맞춘)이 필요하고
2) 여기에 simulator구현, 검증, 그리고 이를 이용한 다양한 실험결과 도출과 이를 통한 discussion까지 들어 있어야 한다. Type 2 PMI, SU-MIMO/MU-MIMO의 성능비교까지
여러 RAN scnario에서의 결과를 얻어 결과를 도출하고 AI/ML CSI compression의 설계화 그 성능, 그리고 그걸 썼을 때의 SU-MIMO/type1과 비교한 성능 개선을 보고, 이건 기존 다른 회사들 결과도 많이 있으니 그거랑 교차 검증하고, MU-MIMO가 type 2랑 비교해 어느정도 개선되는지의 결과까지 어느 정도 얻어야 하고
3) 논문 작성에 chap 1) Introduction, chap 2) MIMO channel model & 5G NR CSI acquisition - scheduling system model, related work (before 기술들)
Chap 3) OAI 기반 RAN twin 개념소개 및 이 가운데 본인이 한 부분 중심으로 (core emulator (RRC와 MAC에 내가 원하는 셋팅을 하기 위한), RAN scenario 구성부분, MAC과 단말에 PMI와 스케줄링 기능부분 등) 구현 제시 및 구현에 대한 기본 검증 결과 제시
Chap 4) 제안하는 AI/ML CSI compression의 개념, 설계, 데이터셋 구성방법 및 모델 자체의 성능평가 ==> 기존 5G (퀄컴 등)에서의 발표결과 대비 단순 compression 성능 (PMI 비트수 대비)이 비슷하면서도 추가적인 특성을 가질 수 있는 것을 제시 ==> 예를 들어 original H와 encoded PMI domain에서의 상대적 distance 유지 등 (즉, decoding 안해봐도 decoding 이후 H들의 distance 예상 가능)이라던지, 또는 covariance matrix나 PDP를 먼저 얻은 후 (이것도 autoencoding하고), 이 결과를 이용해서 H를 autoencoding하면 PMI비트수대비 성능개선을 할 수 있다던지의 두가지 측면에서 try를 해 보도록

Chap 5) 3장에서 만든 시스템 트윈으로 4장 제안하는 것을 써서 RAN scenario에서 다양한 실험결과와 함께 성능분석을 제시
Chap 6) 결론

교수님 코멘트:

여기에 이전 메일에 있는 3개 입력과 출력(core emulator, Env/Radio 환경 및 Mobility 시나리오 입력, Traffic 환경 및  Traffic Emulator)에 대한 제대로 정의하고 거기 맞춰 만드는 것이 필요하다. 그리고 환경/mobility/traffic 시나리오 (준수가 필요한)를 본인이 먼저 설계하고 그걸 twin과 emulator를 통해 구현해 쓸 수 있도록 하는거다.

어떤 환경에서 Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO의 성능 비교를 보는게 의미가 있는건지, 원래부터 준수 논문이 어떤 환경에서 MIMO 성능을 높이려고 하는건지 그게 왜 필요한지, 기술적 한계가 뭔지 등에 대한 구체적인 논리흐름과 내용이 있어야 거기에 맞는 환경에서 실제 실험으로 그 논리에 따른 결과가 잘 나오는지 볼 수 있고 의미도 부여할 수 있고

앞에 이어서 이걸 한다면 무엇을 개선하기 위해 하는 것이고, 기존 AI/ML encoder 의 한계 및 제안하는 거의 다른점과 기대하는 효과가 무엇이고 그게 어떤 상황에서 왜 기대하는 성능이득 효과가 날 것인지에 대한 구체적인 논리가 이미 있어서 그걸 위한 목적성을 가진 설계를 할 수 있어야 하니 이 부분을 위한 PoC는 이미 Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO의 성능 비교에서 원하는 성능의 encoder가 있다고 쳤을 때의 기대성능을 같이 뽑아서 미리 본 이후 그걸 실현하기 위한 설계를 해야 하는거다.

단순히 이런 기술과 이런걸 비교해 봤다가 아니라 제안하는 기술이 왜 중요하고 기존 대비 어떤 효과가 환경마다 어떻게 있는데 그게 기존 기술로는 한계가 있고 내 idea가 novel해서 그걸 해결할 수 있고, 그 해결된 결과가 실제로 매우 중요하다고 본인이 남을 설득할 수 있도록 준비해야 하는 거고
그걸 보여주려면 어떤 Environment와 Radio환경에서 어떤 mobility와 traffic을 가지는 상황에서 기존 방식의 한계가 뭔데 내가 어떻게 극복해서 얼마나 차이를 보여줄 수 있는지에 대한 설계를 준수가 해야 하는거고 이것도 그 차이가 조금 나는 경우, 의미 있게 나는 경우, 최대로 나는 경우 등 최소한 세 가지의 다른 환경 (실제로는 주요 시나리오 parameter가 3개쯤 있다고 하면 3개의 상중하 조합을 전부는 아니더라도 변화에 따른 충분한 결과를 볼 수 있도록 설계가 필요함)

그러니, 우선적으로 어떤 지역에서 기지국이 어떻게 설치되어 있고 단말이 어떻게 분포되어 있고 이동성과 트래픽이 어떻게 셋팅된 상황에서 Type 1 SU-MIMO, Type 2 SU-MIMO, Type 2 MU-MIMO의 성능 비교를 여러 parameter에 따라 비교해 어떤 결과를 얻어 내 주장을 뒷받침하고 연구의 필요성과 중요성을 결과로 보여줄 수 있도록 하라는 얘기다.