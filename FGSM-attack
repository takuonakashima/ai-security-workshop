# FGSM攻撃を行う関数
def fgsm_attack(image, epsilon, data_grad):
    # 勾配（Lossの傾き）の「プラスかマイナスか」の符号(sign)を取得
    sign_data_grad = data_grad.sign()
    
    # 元の画像に、イプシロン(ノイズ強度) × 符号 を足し合わせる
    perturbed_image = image + epsilon * sign_data_grad
    
    # ピクセル値が 0.0 〜 1.0 の範囲に収まるようにクリッピング（はみ出しをカット）
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    # 攻撃用の敵対的サンプル（ノイズが乗った画像）を返す
    return perturbed_image
