# 8-2. Transformer

- Transformer
	- Attention 모델을 극대화시킨 모델 (Attention is All You Need)
	- 기본적으로 Input 시퀀스가 모델에 들어가면 Output 시퀀스가 나오는 건 같다.
	- 다만, 모델 내부에 Encoder와 Decoder를 구성하는 요소가 기존의 RNN 계열로 구성되어 있는 것과 다르다.

- 전반적인 구조
	- Encoding component는 encoder를 stack한 것
	- ![transformer](https://user-images.githubusercontent.com/46666862/80809154-0c251e00-8bfc-11ea-9269-88a96f4f7d9a.PNG)
	- 위 그림과 같이 Encoder가 6개, Decoder 역시 6개로 구성되어 있음. (6개가 최적의 숫자를 의미하는 것은 아님)
	
	- Encoding 블록은 Unmasked, Decoding은 블록은 Masked 방법을 사용
		- Encoder 블록은 Self-Attention과 FF Neural Network 2단계로 이루어짐
		- Decoder 블록은 Masked Self-Attention과 Encoder-Decoder Self-Attention, 그리고 FF NN 3단계로 이루어짐
		- Decoder 블록은 마스킹을 하는데, 뒷 단어가 앞 단어보다 먼저 나오지는 않도록
		
	
- ENCODER Block
	- 인코더들(6개)은 모두 동일한 구조를 가지고 있다. (**하지만 동일한 weight를 가지고 있다는 의미는 아님, 각자의 weight가 존재**)
	
	- Self-Attention Layer와 Feed-forward neural network 2단계로 구성되어 있음
		- 1. Self-Attention Layer: 하나의 토큰(단어)을 처리할 때 함께 주어지는 인풋 시퀀스의 다른 토큰(단어)들을 얼마만큼 중요하게 볼 것인지를 계산하는 Layer
		- 2. FFNN : 아래의 그림처럼 5개의 단어가 각각 Self-Attention Layer를 거친 후 각각의 FFNN 들어간 후 Output을 출력함
			- 즉, 각각의 Network Weight이 존재한다.
		- ![transformer_encoder](https://user-images.githubusercontent.com/46666862/80810050-f3b60300-8bfd-11ea-9981-2b2abb81c523.PNG)
		
	- 두 단계를 6(n)번 반복

- DECODER Block
	- Self-Attention Layer와 Encoder-Decoder Attention Layer와 Feed-forward neural network 3단계로 구성되어 있음
		- 1. Self-Attention Layer
		- 2. Encoder-Decoder Attention Layer: 인코더와 디코더와의 어텐션이 이루어지는 부분
		- 3. FFNN

	- 세 단계를 6(n)번 반복



- 구체적인 구조

	- 1. Input Embedding: Word2Vec과 같은 것들로 토큰을 Embedding (보통 512차원으로)
		- Encoder단의 가장 bottom encoder에서만 Embedding이 일어남
		- 2번째 인코더부터는 이전의 인코더의 아웃풋을 입력으로 받음
		- list의 사이 즈는 Hyper-parameter로 정할 수 있음: 보통은 데이터셋에서 가장 긴 문장의 길이로
		
		
	- 2. Positional Encoding (512차원, 위치정보 반영)
		- LSTM과 같이 input이 시퀀스마다 들어가는 것이 아니라 한 번에 모든 스퀀스가 들어가므로 어떤 단어가 언제 들어갔는지에 대한 정보 손실
		- **각 단어의 위치 정보를 어느정도 복원시켜주기 위한 부분**
		- Input Embedding 벡터와 더해준다 (concat이 아닌 덧셈)
		
		- 의미
			- 1) 해당하는 벡터 인코딩의 크기는 워드임베딩 벡터와 같아야 한다. 그렇게해야 모든 워드들이 같은 방향 또는 같은 크기로 변화한다는 것을 보장할 수 있다.
			- 2) 하나의 문장에서 두 단어의 거리가 멀어지게 되면 둘 사이의 Positional Encoding사이의 거리도 멀어져야 한다.
			
			- ![캡처](https://user-images.githubusercontent.com/46666862/80858973-480dc100-8c98-11ea-9cd8-91147d261ef4.PNG)
			- 위 사진을 보면 거리가 멀어질수록 값이 커지는 경향 존재 (완전히 그렇진 않음)
			- ![transformer_encoding](https://user-images.githubusercontent.com/46666862/80859098-03365a00-8c99-11ea-8acb-118855ec0517.PNG)
			- 위 그림은 100개의(length)시퀀스를 512차원으로 Positional Encoding한 것인데 L2-norm을 보면 평균에 비해 표준편차가 매우 작다.
			- 완벽하진 않지만 꽤 잘 되어있는 것을 알 수 있다.
			
			- **물론 거리가 멀어질수록 값이 커지는 경향이 완벽하진 않으므로 두 토큰 사이가 누가 더 순서가 앞이냐 뒤냐를 알수는 없지만 둘 사이가 얼마나 떨어져있는지는 반영할 수 있다.**
		
	- 3. Multi-Head Attention, Self-Attention Layer (ENCODER단)
		- ![encoder_dependency](https://user-images.githubusercontent.com/46666862/80859645-62966900-8c9d-11ea-9dec-cc1016c72310.PNG)
		- 위 그림처럼 인풋으로 3개의 토큰이 Self-Attention Layer로 들어가서 나오는 아웃풋 Z1, Z2, Z3는 X1, X2, X3 각각에 영향을 받는다. (**토큰들간의 dependency가 존재**)
		- 그러나 Feed-forward Layer에서는 Z1, Z2, Z3가 각각 독립적으로 계산되어 아웃풋을 배출한다. (**토큰들간의 dependency가 존재하지 않음**)  ??
		- 그렇게 배출된 아웃풋은 다음 인코더의 input으로 들어간다.
		
		- Self-Attention이 하는 역할
			- The animal didn't cross the street because **it** was too tired
			- 위와 같은 문장이 있을 때 it이 어떤 단어와 연관되는지를 알아보기 위해 동일한 시퀀스내에 다른 단어를 살펴보는 것
			- The animal과 스코어가 높게 나옴
			
		- Self-Attention in Detail (Single Attention 과정)
			- Step 1: 각각의 인풋 벡터에 대해 세 종류의 벡터를 만듬
				- Query: 현재 보고있는 단어의 Representation한 벡터, 다른 단어를 scoring하기 위한 기준이 되는 값
				- Key: labels와 같은 역할, Query가 주어졌을 때 연관된 다른 단어를 찾을수있도록 해주는 벡터 (자신을 포함한 나머지 것들에 해당)
				- Value: 실제 단어의 Representation 벡터
				- ![key,query,value](https://user-images.githubusercontent.com/46666862/80860450-a17aed80-8ca2-11ea-93e8-73a8efc66834.PNG)
			
				- 각 쿼리, 키, 밸류 벡터는 기존의 Embedding 벡터로부터 만들어짐
				- ![key,query,value,vector](https://user-images.githubusercontent.com/46666862/80860548-35e55000-8ca3-11ea-838d-4f8635afff35.PNG)
				- 위 그림을 보면 Wq, Wk, Wv Matrix가 존재 (이것이 우리가 학습을 통해 구해야 할 미지수, size:(워드임베딩 차원수, 줄이고자 하는 차원 수) )
				- 단어에 대한 임베딩인 X1과 Wq를 matrix 연산 = q1, 마찬가지로 X1와 Wk = k1, X1와 Wv = v1 으로 q1, k1, v1 세 벡터를 만든다.
				- 일반적으로 워드임베딩은 보통 512차원, Q, K, V의 차원은 64로 둔다.  --> 이후에 Multi-Head Attention을 적용하기 위해서
				
			- Step 2: 현재 보고있는 쿼리와 가장 유사한 단어가 무엇이냐를 찾기위해 scoring하는 과정
				- ![key,query,value,vector,step2](https://user-images.githubusercontent.com/46666862/80860687-4649fa80-8ca4-11ea-9cd7-eb4ae4f87412.PNG)
				- 위 그림처럼 30%, 50%의 score를 찾기위해
				
				- ![key,query,value,vector,step2~4](https://user-images.githubusercontent.com/46666862/80860732-b48ebd00-8ca4-11ea-944b-3f15e5ec8c5e.PNG)
				- 위 그림은 score를 구하는 계산 과정
				- q1 벡터와 k1 벡터를 내적 = 112, q2 벡터와 k2 벡터를 내적 = 96 (Thinking을 기준으로)
				- **Machines을 기준으로 하면 k2 * q1, k2 * q2를 softmax해서 계산하면 됨 (z2 구하는 법)**
				
			- Step 3: score / root(차원수)를 해줌
				- 위에서는 Root(64) = 8이므로, 112/8=14, 96/8=12
				- 이는 gradients가 stable해진다는 효과가 있음.
				
			- Step 4: Softmax 연산
				- 그러면 14는 0.88, 12는 0.12가 됨
				- 이 스코어가 의미하는 것은 현재 보고있는 단어에 해당하는 position 단어가 얼마나 중요한 역할을 하는지를 표현
				- 즉, Thinking이 지금 보고 있는 단어에 0.88만큼 중요하다.
				
			- Step 5: Softmax로 연산된 score와 value벡터를 Multiply
				- ![key,query,value,vector,step5~6](https://user-images.githubusercontent.com/46666862/80860892-c6bd2b00-8ca5-11ea-9f42-79388284a24a.PNG)
				- 위 그림으로 보면 v1 벡터와 0.88을 곱하고, v2 벡터와 0.12를 곱한다.
				
			- Step 6: Step5 벡터를 Sum
				- z1 = 0.88v1 + 0.12v2
				- 최종적으로 z1을 첫 번째 단어의 self-attention layer의 output으로 사용하겠다.
				- ![key,query,value,vector,step6](https://user-images.githubusercontent.com/46666862/80860977-7a261f80-8ca6-11ea-88ad-142a7cd766ff.PNG)
				
				
			- Matrix calculation
				- ![self-attention_matrix_cal](https://user-images.githubusercontent.com/46666862/80861043-f4ef3a80-8ca6-11ea-8013-266013a8067f.PNG)
				- 위 그림과 같이 matrix형태로 계산되는 것을 볼 수 있다.
				- X벡터 (2x4 = 문장내 단어 수 x 워드임베딩 차원 수)
				- Q, K, V벡터 (2x3 = 문장내 단어 수 x 줄이려는 차원 수)
				
				- 이후, Q와 K(transpose)를 matrix연산을 하면 2x2 matrix가 나옴
				- 2x2 matrix의 1행 1열은 첫 번째 단어(Q벡터)와 첫 번째 단어(K벡터)간의 score, 즉 q1*k1, q1*k2, q2*k1, q2*k2가 각 요소로 들어가 2x2 matrix를 구성
				
				
			- 과정을 다룬 다른 그림
				- ![gpt2_step1](https://user-images.githubusercontent.com/46666862/80861402-50222c80-8ca9-11ea-853e-55059cf4e62a.PNG)
				- ![gpt2_step2](https://user-images.githubusercontent.com/46666862/80861403-51535980-8ca9-11ea-8512-f786666b27e0.PNG)
				- ![gpt2_step3](https://user-images.githubusercontent.com/46666862/80861404-51ebf000-8ca9-11ea-96da-54c8b295d01e.PNG)
				- ![gpt2_total](https://user-images.githubusercontent.com/46666862/80861459-ad1de280-8ca9-11ea-9646-e7a4f6308514.PNG)
				- 최종적으로 X1, X2, X3, X4 각각을 이용해서 Z1, Z2, Z3, Z4를 만듬
				
				
		- Multi-head Attention
			- 위의 과정은 Single Attention의 과정
			- 예를들어, it이 어떤 단어를 볼지를 하나의 경우의 수만 허용하는 것이 아닌 여러 경우의 수를 보겠다는 것
			- Single Attention을 전부 다르게 8개 각각을 따로 해서 만들어낸 z값들을 사용하자
			
			계산과정
				- ![multihead](https://user-images.githubusercontent.com/46666862/80861799-924c6d80-8cab-11ea-8cc5-cc862729cd30.PNG)
				- 1) 각 Single Attention(Attention Head)으로 나온 z0~z7까지 8개를 concat
				- 2) WO라는 Weight matrix와 행렬곱, WO의 size는 (z0~z7을 concat한 벡터의 열 크기, 워드임베딩 벡터 사이즈) = 보통(512, 512)
					- WO 역시 Wq, Wk, Wv Matrix 처럼 학습하는 weight
				- 3) 원래 인풋임베딩과 동일한 차원의 Self-Attention의 output을 만들어냄
				
				
				- ![multihead-attention](https://user-images.githubusercontent.com/46666862/80861850-1f8fc200-8cac-11ea-9134-480c8466a2db.PNG)
				- 위 그림은 Multi-Head Attention의 종합적인 과정
				
				
	- 4. Add(Residual Connection), Normalization
		- Residual Block: f(x) + x, 미분을 하게되면 f'(x) + 1이 되는데 f'(x)가 매우 작은 값이 나오더라도 1만큼 흘려주므로 학습에서 유리
		- self-attention을 통해 나온 Z값에 원래의 워드임베딩 벡터 X를 더해주고 LayerNorm을 해준다.
		- ![residual](https://user-images.githubusercontent.com/46666862/80862115-819cf700-8cad-11ea-9061-b37535592389.PNG)
		
		- 아래 그림을 보면 Residual & Norm을 여러 번 해주는 것을 볼 수 있다.
		- ![residual_many](https://user-images.githubusercontent.com/46666862/80862164-cfb1fa80-8cad-11ea-830d-59b57221f933.PNG)
		
	- 5. Position-wise Feed-Forward Networks
		- Self-Attention과 Residual & Norm Layer를 통과한 벡터들을 Feed Forward에 넣어준다.
		- Fully Connected Network임.
		- **하나의 인코더 내에서는 각각의 z1, z2는 Weight를 공유함**
		- **단, 첫 번째 인코더와 두 번째 인코더는 weight를 공유하지 않음**
		- ![ffnn](https://user-images.githubusercontent.com/46666862/80862271-ae9dd980-8cae-11ea-9094-175cf9062caa.PNG)

		- 하나의 인코더 내에서 z1, z2, ... zn FFNN의 구조가 같다(두 번째 인코더는 다른 FFNN의 구조를 가질 수 있음)
		- 위 그림처럼 첫 번째 인코더는 2048차원의 FFNN 이지만, 두 번째 인코더는 1024차원의 FFNN일 수 있다.
		
		- Convolution 연산을 활용하면 FFNN의 계산을 더 빠르게 할 수 있다.
		
		
	- 6. Masked Multi-Head Attention (DECODER단)
		- 디코더 단에서 Self-Attention Layer는 자기 자신보다 앞에 있는 단어에만 attention score를 매길수있다.
		- ![masked_attention](https://user-images.githubusercontent.com/46666862/80862548-e574ef00-8cb0-11ea-8465-0906db447662.PNG)
		- 위 그림에서 인코더 단에서 Thinking은 뒤에 나오는 Machines을 고려할 수 있지만 디코더 단에서는 볼 수 없다.
		- 따라서, q1 x k2 같은 경우는 마스킹을 해서 -inf 값을 주어 처리함.
		- 디코더에서는 자신보다 앞에 나온 단어만 attention에 사용할 수 있다.
		- ![masked_attention_2](https://user-images.githubusercontent.com/46666862/80862639-8a8fc780-8cb1-11ea-9ce6-6107aa45335c.PNG)
		- 위 그림처럼, 실제로 시퀀셜하게 할 필요없이 한 번에 계산을 하고 그 부분만 마스킹을 해주면 된다.

	- 7. Multi-Head Attention with Encoder Outputs
		- 인코더의 아웃풋(K, V 벡터)과 디코더 사이의 Multi-Head Attention
		- 인코더의 아웃풋이 디코더 각각을 Self-Attention할 때 계속 사용됨.
		- 디코더에서는 마스킹때문에 시퀀셜하게 작동됨.
		- ![decoder_attention](https://user-images.githubusercontent.com/46666862/80862802-c5dec600-8cb2-11ea-9d57-42f8d143ba47.PNG)
		

	- 8. Final Linear and Softmax Layer
		- 단순 Linear 및 Softmax
		- ![final_step](https://user-images.githubusercontent.com/46666862/80862803-c70ff300-8cb2-11ea-883e-4e3284f6db98.PNG)
		- FCN(Linear)를 거친 후 vocab_size만큼의 벡터를 softmax하면 가장 확률이 높은 위치의 index를 알 수 있음
		- 그 index에 해당하는 단어를 Return해주는 부분
		
- BLEU score
	- ![BLEU](https://user-images.githubusercontent.com/46666862/80862894-6d5bf880-8cb3-11ea-8461-a1a7a6b4eb34.PNG)
	- ![bleu_ex](https://user-images.githubusercontent.com/46666862/80862896-6e8d2580-8cb3-11ea-8a7a-bc244853758e.PNG)

		
- Transformer의 Hyper-Parameter
	- ![transformer_parameter](https://user-images.githubusercontent.com/46666862/80862927-aac08600-8cb3-11ea-89b9-353d9623544b.PNG)
	- 네모 박스친 부분이 모두 Hyper-Parameter
	- 설정할 부분이 너무 많아서 요즘에는 Task와 목적만 정해주면 최적의 모델을 찾아주는 AutoML을 사용하려고 함.



일반 Seq2Seq
	- 인풋으로 여러개의 시퀀스가 들어가면 아웃풋으로 여러개의 시퀀스가 출력되는 모델
	- 인코더와 디코더 두개의 단으로 구성되어 있고, 각 구조는 RNN 계열의 모델을 사용
	- 인코더 단에서 최종적으로 출력하는 Hidden state 벡터가 Context 벡터임
	- 인코더 단에서 출력된 Hidden state 벡터만을 사용해 디코더 단에서 아웃풋 시퀀스를 계산함

Attention
	- 인코더 단의 시퀀스를 거치며 출력되는 Hidden State 벡터를 모두 사용하자는 것이 Main idea
	- 모든 Hidden state 벡터(h1, h2, h3)를 디코더 단에 넘겨주어 디코더 단에서 각 시퀀스마다 출력되는 hidden state 벡터와 내적을 해줌
	- 즉, dot(h1, h4), dot(h2, h4), dot(h3, h4)를 하면 각각이 스칼라이고 연결하면 1x3의 벡터가 나온다.
	- 그 벡터를 softmax를 통해 합이 1로 만들어준다, [0.96, 0.02, 0.02] (attention score를 계산하는 과정)
	- 이후 attention score 벡터와 h1,h2,h3를 각각 가중합을 해준다. 0.96 x h1 + 0.02 x h2 + 0.02 x h3
	- 위의 가중합을 한 벡터가 context vector가 되고 context벡터와 h4벡터를 컨캣해주어 1x8의 벡터를 만든다
	- 1x8의 벡터를 Feed-forward Neural network에 넣어주어 1x4(n)의 Output 벡터를 만든다.


