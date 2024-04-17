## Soda Vision Inferencia JS API

### Para realizar a inferencia do Soda Vision, com um modelo treinado Local, com uma API JS, siga os seguintes passos:

- Clone este repositório;  
- Descompate o seu modelo treinado localmente;  
- Coloque os arquivos do seu modelo dentro da pasta:  
  - soda_vision_model/  
- Rode os seguintes comandos:  
  - ````docker
    docker build --no-cache -t sv-inferencia:latest .
    ````

  - ````docker
    docker run --name sv-inferencia -dp 127.0.0.1:3000:3000 sv-inferencia
    ````

Após estes comando sua aplicação deve estar rodando normalmemente, para confirmar digite:

  - ```docker
    docker ps
    ```

E verifique se o container sv-inferencia está ativo.

Rota:

  - ```http
    http://localhost:3000/api-docs
    ```

<hr>

## Documentação API

#### Rota de uso

````http
POST /classification-image
````

| Parâmetro   | Tipo       | Descrição                           |
| :---------- | :--------- | :---------------------------------- |
| `data` | `arquivo-base64` | **Obrigatório**. Imagem para classificação |

#### Retorno
````
{
  "classification": "OK",
  "confidence-score": 99
}
````


## Exemplos Python via script

```python
import requests
import cv2
import base64

def send_image_as_base64(image_path, url):
    """Envia uma imagem codificada em base64 para o servidor."""
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')  # Codifica para base64 e depois decodifica para string
    headers = {'Content-Type': 'text/plain'}
    response = requests.post(url, data=image_base64, headers=headers)
    return response.json()

if __name__ == "__main__":
    url = 'http://localhost:3000/classification-image'
    image_path = 'image.jpg'
    result = send_image_as_base64(image_path, url)
    print(result)
```