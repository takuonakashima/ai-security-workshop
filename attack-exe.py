# 攻撃の強さ
epsilon = 0.1

# 画像に対する勾配を計算できるように設定
images.requires_grad = True

# 1. まず普通にAIに推論させる
output = model(images)
loss = F.nll_loss(output, labels) # 正解とのズレ（Loss）を計算

# 2. AIのモデル全体を「逆伝播」させ、画像に対する「Lossの傾き（勾配）」を計算する
model.zero_grad()
loss.backward()

# 計算された画像に対する勾配データを取得
data_grad = images.grad.data

# 3. FGSM攻撃関数を呼び出し、敵対的サンプルを作成！
perturbed_data = fgsm_attack(images, epsilon, data_grad)

# 4. 作成した毒入り画像（敵対的サンプル）を、もう一度AIに入力する
output_adv = model(perturbed_data)
max_prob_adv, predicted_adv = torch.max(F.softmax(output_adv, dim=1), 1)

# 結果の表示
print(f"【攻撃後】AIの予測: {predicted_adv.item()} (確信度: {max_prob_adv.item()*100:.2f}%)")
print(f"本当の正解: {labels[0].item()}")

# 攻撃画像の表示
plt.imshow(perturbed_data[0].detach().cpu().numpy().squeeze(), cmap='gray')
plt.title(f"Adversarial Example (Predicted: {predicted_adv.item()})")
plt.axis('off')
plt.show()
