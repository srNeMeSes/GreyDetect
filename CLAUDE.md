# CLAUDE.md — GreyDetect

Referência principal de arquitetura, implementação, organização, testes e validação do **GreyDetect**: uma rede neural convolucional **autoral** para detecção de objetos em tempo real, escrita em C++ com LibTorch e OpenCV. O primeiro alvo é a detecção de **pessoas**.

Todo o conteúdo do projeto — documentação, comentários de código, mensagens de log, mensagens de commit, relatórios e instruções — deve ser escrito em **português brasileiro (pt-BR)**. Identificadores de código (nomes de classes, funções e variáveis) podem seguir a convenção em inglês, desde que adotada de forma consistente (ver §9).

---

## 1. Princípios inegociáveis

Estes três princípios têm precedência sobre qualquer outra instrução deste documento. Em caso de conflito, eles vencem.

1. **Primeiro, correção matemática; depois, velocidade.**
   Uma rede rápida que não aprende corretamente não é um detector funcional. Nenhuma otimização de desempenho é feita antes de o pipeline estar matematicamente correto e validado (Fases 3–7). Toda otimização posterior deve preservar os resultados numéricos validados, dentro de uma tolerância declarada.

2. **Cada etapa precisa produzir evidências.**
   Compilação, testes, gradientes, redução da perda, métricas de detecção e benchmarks devem ser verificáveis e reproduzíveis: comando executado, saída real, hardware, versões e configuração. O que não foi medido não é afirmado. O que não pôde ser validado é registrado como **não validado**, com o motivo.

3. **A arquitetura é autoral.**
   A LibTorch fornece a infraestrutura (tensores, autograd, camadas primitivas, otimizadores). A arquitetura, as decisões de projeto, a atribuição de alvos, as funções de perda específicas do detector e os experimentos pertencem ao autor do projeto. O YOLOv8 é **apenas referência conceitual**: é proibido usar sua implementação, seus pesos pré-treinados ou qualquer detector de terceiros como substituto da arquitetura autoral.

---

## 2. Visão geral

| Item | Definição |
|---|---|
| Tipo de modelo | Detector de estágio único (*single-stage*), inspirado conceitualmente em detectores modernos (YOLOv8) |
| Alvo da v1 | Uma classe: `pessoa`, usando as caixas de **corpo inteiro** (`fbox`) do CrowdHuman |
| Saída | Bounding boxes + escore de confiança, após pós-processamento (NMS ou alternativa justificada) |
| Evoluções futuras | Cabeça (`hbox`), corpo visível (`vbox`), rosto — cada uma exige definição explícita de classes e alvos |
| Uso | Imagens estáticas e vídeo (arquivo ou câmera) via OpenCV |

"Tempo real" só pode ser afirmado com um **objetivo de desempenho numérico definido** (ex.: FPS mínimo em resolução X, em hardware Y) e **benchmarks reproduzíveis** que o sustentem. Não se presume que a v1 atingirá esse objetivo.

---

## 3. Ambiente de desenvolvimento

### 3.1 Estado verificado (inspeção de 2026-10-15)

| Componente | Situação encontrada |
|---|---|
| Sistema operacional | Windows 11 Pro 10.0.26200 |
| CPU | Intel Core i3-10100F — 4 núcleos / 8 threads, 3,6 GHz |
| RAM | 15,9 GB |
| GPU | NVIDIA GeForce GT 730 (GK208, **Kepler, compute capability 3.5**), driver 30.0.14.7298 |
| Disco livre (C:) | ~78 GB |
| Compilador C++ | **Apenas MinGW-w64 g++ 15.2.0**. Pastas do Visual Studio existem, mas **não há toolset MSVC instalado** |
| CMake | 4.1.0 |
| Geradores | Ninja e mingw32-make (em `C:\MinGW\bin`) |
| Git | 2.52.0 |
| LibTorch | **Não instalado** |
| OpenCV | **Não instalado** |
| CUDA Toolkit / `nvcc` / `nvidia-smi` | **Não encontrados** |
| Python | 3.13.7 (sem PyTorch) |
| GitHub CLI (`gh`) | Não instalado |

### 3.2 Bloqueios e riscos identificados

1. **LibTorch × MinGW.** Os binários oficiais da LibTorch para Windows são compilados com MSVC e não são compatíveis em ABI com MinGW. Usar MinGW exigiria compilar a LibTorch a partir do código-fonte, com adaptações não oficiais. **Caminho recomendado:** instalar o Visual Studio Build Tools (toolset MSVC + Windows SDK) e usar a LibTorch oficial. Decisão pendente na Fase 0.
2. **Debug × Release no Windows.** A documentação da LibTorch informa que builds Debug e Release não são compatíveis em ABI no Windows. A configuração do projeto deve casar com a variante da LibTorch baixada.
3. **CUDA inviável na GPU atual.** A GT 730 (Kepler, CC 3.5) não é suportada pelos binários CUDA atuais do PyTorch/LibTorch (que partem de CC 5.0 ou superior, conforme a versão). O desenvolvimento deve ser tratado como **somente CPU** até que haja hardware compatível. O código deve manter seleção de dispositivo (`torch::kCPU` / `torch::kCUDA`) com detecção em tempo de execução, mas o caminho CUDA **não pode ser declarado validado** sem execução real em GPU suportada.
4. **Treinamento em CPU.** Treinar no CrowdHuman completo em um i3 de 4 núcleos será lento. Isso impacta o tamanho do modelo, a resolução de entrada e o número de épocas. Experimentos devem ser dimensionados para esse hardware, e qualquer extrapolação deve ser rotulada como tal.
5. **CMake 4.x.** Removeu a compatibilidade com `cmake_minimum_required` abaixo de 3.5. Verificar se os arquivos de configuração da LibTorch e do OpenCV escolhidos funcionam com essa versão antes de fixá-la.

### 3.3 Versões fixadas

> **A DEFINIR na Fase 0.** Preencher após verificar a compatibilidade real entre compilador, LibTorch, OpenCV, CMake e framework de testes. Registrar aqui a versão exata, a origem (URL oficial) e a variante (CPU/CUDA, Debug/Release).

| Componente | Versão | Origem | Observações |
|---|---|---|---|
| Padrão C++ | — | — | — |
| Compilador | — | — | — |
| LibTorch | — | — | — |
| OpenCV | — | — | — |
| Framework de testes | — | — | GoogleTest ou Catch2 |
| CMake | — | — | — |

Regras para dependências:

- Não inventar APIs, funções, parâmetros ou recursos de bibliotecas. Na dúvida, consultar a documentação oficial da versão fixada (cabeçalhos da LibTorch incluídos).
- Não adicionar dependências sem justificar a necessidade e registrar aqui.
- Bibliotecas de terceiros grandes (LibTorch, OpenCV) **não são versionadas** no repositório; o caminho delas é passado ao CMake (ex.: `CMAKE_PREFIX_PATH`).

---

## 4. Arquitetura neural

### 4.1 Proposta inicial (não obrigatória)

| Componente | Papel |
|---|---|
| **Backbone** | Extração hierárquica de características por convoluções e blocos neurais |
| **Neck** | Fusão de características entre escalas (pode ser omitido na v1) |
| **Detection Head** | Previsão das coordenadas das caixas e do escore de objeto/classe |
| **Pós-processamento** | Decodificação, limiar de confiança e NMS (ou alternativa tecnicamente justificada) |

Regras:

- A **primeira versão funcional deve ser tão simples quanto possível**: por exemplo, uma única escala de saída, sem neck. A complexidade só aumenta quando um experimento mostrar a necessidade.
- Cada aumento de complexidade exige uma **hipótese técnica** e um **método de avaliação** (qual métrica deve melhorar e em qual conjunto de dados).
- Proibido copiar código, configurações ou pesos do YOLOv8 ou de outros detectores. Conceitos publicados (convolução, normalização em lote, FPN, perdas baseadas em IoU etc.) podem ser usados, desde que implementados de forma autoral e com a referência citada em `docs/`.

### 4.2 Documentação obrigatória da arquitetura

Manter em `docs/arquitetura.md`, atualizado a cada mudança estrutural:

- Diagrama ou tabela do fluxo de dados, camada por camada.
- **Dimensões dos tensores** em cada estágio, no formato `[N, C, H, W]`.
- Stride de cada saída em relação à entrada.
- Quantidade de parâmetros por módulo e total (medida pelo código, não estimada à mão).
- Funções de ativação, normalizações e inicialização dos pesos.
- Representação das caixas previstas e sua decodificação.
- Estratégia de atribuição de alvos (*target assignment*).
- Formulação matemática das funções de perda.
- Justificativa de cada decisão relevante e alternativas descartadas.

### 4.3 Convenções numéricas (fixar na Fase 4 e não alterar sem migração explícita)

- Formato interno das caixas: **`[x1, y1, x2, y2]` em pixels da imagem de entrada da rede**, com origem no canto superior esquerdo, `x` para a direita e `y` para baixo. As anotações do CrowdHuman (`[x, y, w, h]`) são convertidas na leitura.
- Toda conversão de coordenadas (letterbox/redimensionamento ↔ imagem original) passa por funções únicas e testadas. Nada de conversões duplicadas espalhadas pelo código.
- Tipo padrão dos tensores de treino: `float32`. Qualquer uso de precisão reduzida é otimização da Fase 9.

---

## 5. Dataset: CrowdHuman

### 5.1 Licenciamento (verificado em 2026-10-15 em crowdhuman.org/download.html)

Termos oficiais, resumidos:

- Uso **somente para pesquisa e educação não comerciais**.
- **Proibido redistribuir as imagens.**
- A Megvii Technology não oferece garantias; o usuário assume total responsabilidade pelo uso.

Consequências para o projeto:

- **Nenhuma imagem ou anotação do CrowdHuman pode ser versionada no Git** (o repositório é público). O diretório de dados fica fora do controle de versão (`.gitignore`).
- Pesos treinados com o CrowdHuman herdam a restrição de uso não comercial. Não presumir autorização para uso comercial do modelo.
- Ao citar resultados, citar o artigo do CrowdHuman, conforme solicitado pelos autores.

### 5.2 Arquivos e formato

Arquivos oficiais: `CrowdHuman_train01.zip`, `CrowdHuman_train02.zip`, `CrowdHuman_train03.zip`, `CrowdHuman_val.zip`, `CrowdHuman_test.zip`, `annotation_train.odgt`, `annotation_val.odgt`.

Tamanhos oficiais: 15.000 imagens de treino, 4.370 de validação, 5.000 de teste. **O conjunto de teste não possui anotações públicas.**

Formato `.odgt`: um objeto JSON por linha, um por imagem.

- `ID`: nome da imagem.
- `gtboxes`: lista de objetos com:
  - `tag`: `"person"` ou `"mask"` (`"mask"` marca regiões a ignorar, como multidões e reflexos).
  - `fbox` (corpo inteiro), `vbox` (corpo visível), `hbox` (cabeça): `[x, y, w, h]`.
  - `extra` (opcional): pode conter `ignore` (0/1), `box_id`, `occ`.
  - `head_attr` (opcional): pode conter `ignore` (0/1), `unsure`, `occ`.
- As chaves de `extra` e `head_attr` são opcionais: o leitor deve tratar a ausência sem falhar.

Validar esses detalhes contra os arquivos reais na Fase 2. Se divergirem, os arquivos reais prevalecem e este documento é corrigido.

### 5.3 Decisões da v1

- **Alvo:** `fbox` de instâncias com `tag == "person"`.
- **Caixas ignoradas:** instâncias com `tag == "mask"` ou `extra.ignore == 1` não são alvos positivos. Como serão tratadas na perda (excluídas da atribuição ou marcadas como regiões neutras) é decisão da Fase 4, que deve ser documentada.
- **Caixas fora da imagem:** `fbox` pode ultrapassar os limites da imagem (corpo cortado pelo quadro). A política (recortar aos limites, descartar abaixo de uma área mínima) deve ser definida, testada e registrada, com a contagem de caixas afetadas.
- `vbox` e `hbox` são lidos e preservados na estrutura interna, mas **não** são usados como alvo na v1.

### 5.4 Divisão dos dados

- **Treino:** subconjunto de `annotation_train.odgt`.
- **Validação interna** (seleção de hiperparâmetros, *early stopping*): subconjunto fixo separado do treino oficial, com semente e lista de IDs registradas em arquivo.
- **Teste** (avaliação final): o **val oficial** (4.370 imagens), usado **somente** para relatórios de avaliação e nunca para ajustar hiperparâmetros.
- Verificação automática de vazamento: nenhum `ID` pode aparecer em mais de um conjunto.

### 5.5 Pipeline de dados

- Download manual, conforme os termos oficiais; o projeto apenas documenta onde colocar os arquivos.
- Leitura e validação das anotações: JSON malformado, caixas com largura/altura ≤ 0, coordenadas não finitas, imagens ausentes ou corrompidas.
- Relatório de qualidade dos dados: contagens de imagens, instâncias, caixas ignoradas, caixas recortadas e descartadas.
- Pré-processamento: redimensionamento (letterbox ou outra estratégia documentada), normalização e conversão `cv::Mat` → `torch::Tensor`, com testes que verifiquem a ordem dos canais (BGR/RGB) e o layout (HWC → CHW).
- *Data augmentation*: somente depois de o pipeline básico estar validado. Toda transformação geométrica deve transformar as caixas junto e ter teste dedicado.
- Inspeção visual: ferramenta que salva imagens com as caixas anotadas desenhadas, para conferência humana.

---

## 6. Regra fundamental: desenvolvimento incremental com validação obrigatória

**É proibido implementar a rede, o treinamento e a inferência em uma única etapa.** O projeto avança por etapas pequenas, independentes, verificáveis e documentadas.

### 6.1 Antes de cada etapa

1. Identificar o objetivo específico.
2. Inspecionar os módulos e as dependências existentes.
3. Listar os arquivos que serão criados ou modificados.
4. Definir critérios de aceitação **objetivos e verificáveis**.
5. Identificar riscos técnicos e dependências de etapas anteriores.

### 6.2 Durante a etapa

1. Modificar apenas o necessário para o objetivo da etapa.
2. Não alterar módulos já validados sem justificativa explícita.
3. Criar testes unitários e de integração adequados.
4. Compilar após mudanças relevantes.
5. Executar os testes aplicáveis.
6. Investigar e corrigir erros antes de prosseguir.

### 6.3 Ao concluir cada etapa: relatório obrigatório

```markdown
## Relatório — Fase X, Etapa Y: <título>

**Implementado:** ...
**Arquivos criados/modificados:** ...
**Comandos executados:** (copiar os comandos reais)
**Resultados reais:** (saída resumida dos testes/compilação; números medidos)
**Ambiente:** (SO, compilador, LibTorch, OpenCV, dispositivo CPU/GPU)
**Critérios de aceitação:** [x] atendido / [ ] não atendido — com evidência
**Não validado / limitações:** ...
**Problemas pendentes:** ...
**Próxima etapa recomendada:** ...
```

### 6.4 Regras de evidência

- **Não declarar uma etapa concluída sem evidência de validação.**
- **Não inventar** resultados de compilação, testes, treinamento, métricas ou benchmarks.
- Se um teste falhar: **parar** o avanço para etapas dependentes, investigar a causa raiz e corrigir antes de continuar. Não desabilitar nem enfraquecer testes para fazê-los passar.
- Se o hardware ou o ambiente impedirem uma validação (ex.: CUDA nesta máquina), registrar a limitação explicitamente e não apresentar o resultado como comprovado.

---

## 7. Plano de desenvolvimento por fases

A ordem pode ser ajustada conforme as dependências técnicas, mas nenhuma fase começa antes de os critérios de aceitação das fases das quais depende estarem atendidos.

### Fase 0 — Análise e planejamento
- Inspecionar o ambiente (compilador, SO, LibTorch, OpenCV, CUDA). *Feito em 2026-10-15; ver §3.1.*
- Resolver os bloqueios da §3.2 (toolchain e variante da LibTorch).
- Definir as versões (§3.3) e a estrutura de diretórios (§9).
- Registrar as decisões em `docs/decisoes.md`.
- **Aceitação:** versões fixadas e justificadas; estrutura aprovada pelo autor; plano da Fase 1 com critérios objetivos.

### Fase 1 — Infraestrutura e compilação
- Estrutura do projeto, `CMakeLists.txt`, integração com LibTorch, OpenCV e o framework de testes.
- Executável mínimo que imprime as versões das bibliotecas e os dispositivos disponíveis.
- Um teste trivial executado pelo CTest.
- **Aceitação:** build limpo a partir de um diretório vazio, com comandos documentados; o executável roda; um tensor é criado em CPU; uma imagem é lida pelo OpenCV; os testes passam. Disponibilidade de CUDA reportada como verdadeira ou falsa, sem falhar.

### Fase 2 — Pipeline de dados
- Leitor de `.odgt`, conversão e validação de caixas, pré-processamento, divisão dos dados, relatório de qualidade, ferramenta de visualização.
- **Aceitação:** testes de parsing (incluindo campos opcionais ausentes e entradas inválidas); testes de ida e volta das conversões de coordenadas; verificação de vazamento entre conjuntos; amostra visual conferida pelo autor.

### Fase 3 — Componentes neurais fundamentais
- Primeiros módulos convolucionais (`torch::nn::Module`).
- **Aceitação:** formas de saída corretas para várias entradas; gradientes finitos e não nulos em todos os parâmetros; comparação com casos de referência calculados à mão ou com verificação numérica de gradiente (diferenças finitas) em tensores pequenos, com tolerância declarada.

### Fase 4 — Primeira arquitetura funcional
- Backbone inicial, cabeça de detecção simples, representação e decodificação das caixas, atribuição de alvos, perdas de localização e de objeto/classe.
- **Aceitação:** testes com entradas sintéticas; perda igual a zero (ou mínima conhecida) quando a previsão é perfeita; atribuição de alvos verificada em casos construídos à mão; contagem de parâmetros documentada.

### Fase 5 — Validação do treinamento
- Propagação direta, retropropagação, gradientes finitos, parâmetros efetivamente atualizados.
- **Sobreajuste deliberado** em um conjunto minúsculo (ex.: 1 a 8 imagens).
- **Aceitação:** a perda cai de forma consistente no conjunto minúsculo; o modelo reproduz as caixas dessas imagens com IoU alto (limiar a definir); sem NaN/Inf. **Não avançar sem essa evidência.**

### Fase 6 — Treinamento experimental
- Carregamento em lotes, otimizador, agenda de taxa de aprendizado, checkpoints e retomada, sementes fixas, registro de perdas, métricas e configuração completa de cada experimento.
- **Aceitação:** um treino interrompido e retomado do checkpoint continua de forma coerente; logs permitem reproduzir o experimento; avaliação feita na validação interna, que não atualiza pesos.

### Fase 7 — Avaliação objetiva
- Precisão, recall, IoU, AP e mAP, com critérios documentados (limiar de IoU, tratamento de caixas ignoradas, ponto de operação).
- Latência por imagem e FPS em condições controladas; uso de memória.
- O código de métricas deve ser testado contra casos com resultado conhecido.
- **Aceitação:** relatório de avaliação no conjunto de teste (§5.4), com hardware, versões e configuração. Caixas "visualmente plausíveis" não são critério de sucesso.

### Fase 8 — Inferência em tempo real
- Captura de vídeo (OpenCV), pré-processamento, inferência, pós-processamento, desenho de caixas e escores.
- Medir separadamente: latência da rede isolada **e** latência de ponta a ponta (captura → exibição).
- **Aceitação:** funciona com arquivo de vídeo e com câmera; FPS e latência reportados com hardware identificado; nenhuma confusão entre a velocidade da rede e a da aplicação.

### Fase 9 — Otimização e evolução
- Análise de gargalos (com *profiling*, não suposição), fusão multiescala, otimizações de memória e execução, comparação controlada de variantes, planejamento de `hbox`/`vbox`/rosto.
- **Aceitação:** cada mudança acompanhada da hipótese, do experimento e do resultado comparado à linha de base; otimizações não degradam as métricas além da tolerância declarada.

---

## 8. Testes

- Framework: GoogleTest ou Catch2 (escolha na Fase 0), integrado ao CTest.
- Os testes não dependem do dataset completo. Usar *fixtures* pequenas e sintéticas geradas pelo próprio teste ou arquivos mínimos criados pelo projeto (nunca imagens do CrowdHuman versionadas).
- Testes que exigem CUDA são condicionais e reportam "pulado" quando não houver GPU suportada, nunca "aprovado".
- Comparações de ponto flutuante usam tolerâncias explícitas (`torch::allclose` com `rtol`/`atol` declarados).
- Sementes fixas em todo teste com aleatoriedade.
- Categorias mínimas: dados e transformações; formas de tensores; gradientes; perdas; atribuição de alvos; decodificação e NMS; métricas; checkpoint (salvar → carregar → mesmo resultado).

---

## 9. Organização e qualidade do código

### 9.1 Estrutura proposta (a aprovar na Fase 0)

```
GreyDetect/
├── CLAUDE.md
├── CMakeLists.txt
├── README.md
├── .gitignore
├── cmake/                 # módulos auxiliares do CMake
├── configs/               # configurações de experimentos (texto)
├── docs/
│   ├── arquitetura.md     # fluxo, tensores, parâmetros, perdas
│   ├── decisoes.md        # registro de decisões (data, contexto, decisão, alternativas)
│   ├── dados.md           # pipeline e relatório de qualidade do CrowdHuman
│   └── experimentos/      # um relatório por experimento
├── include/greydetect/    # cabeçalhos públicos, por módulo
├── src/
│   ├── core/              # configuração, dispositivo, logging, utilitários
│   ├── data/              # leitura do CrowdHuman, transformações, dataset
│   ├── model/             # backbone, neck, head
│   ├── loss/              # perdas e atribuição de alvos
│   ├── train/             # laço de treino, checkpoints
│   ├── eval/              # métricas e avaliação
│   ├── infer/             # inferência, decodificação, NMS
│   └── app/               # executáveis (treino, avaliação, vídeo, visualização)
├── tests/                 # espelha src/
├── scripts/               # utilitários de apoio (sem lógica do modelo)
├── data/                  # NÃO versionado (CrowdHuman)
└── runs/                  # NÃO versionado (checkpoints, logs)
```

### 9.2 Diretrizes

- Responsabilidades bem definidas por módulo; arquivos pequenos e coesos.
- Sem duplicação de código; sem abstrações prematuras; sem dependências desnecessárias.
- Erros tratados explicitamente (exceções com mensagens claras em pt-BR ou retornos verificados); nunca ignorar falhas de leitura de arquivo, imagem ou checkpoint.
- Validação de formas de tensores nas fronteiras dos módulos (`TORCH_CHECK` ou equivalente) com mensagens que informem a forma esperada e a recebida.
- Código compatível com o padrão C++ fixado na §3.3.
- Comentários explicam o **porquê** e a matemática relevante; não repetem o que o código já diz.
- Convenção de nomes: definir na Fase 1 e registrar aqui.

---

## 10. Regras para o desenvolvimento assistido por IA

- Não gerar o projeto inteiro de uma vez; trabalhar em entregas pequenas e verificáveis.
- Inspecionar o código existente antes de modificá-lo.
- Não reescrever módulos validados sem justificativa.
- Explicar as decisões matemáticas e arquiteturais importantes.
- Não ocultar limitações ou incertezas.
- Não substituir componentes autorais por bibliotecas de detecção prontas sem autorização explícita do autor.
- Usar bibliotecas para infraestrutura e operações matemáticas, preservando o caráter autoral da arquitetura.
- Não afirmar que o modelo é de tempo real sem medições.
- Não afirmar que o treinamento convergiu sem examinar curvas e métricas.
- Não inventar benchmarks, métricas ou evidências.
- Manter este documento e `docs/` atualizados conforme o projeto evolui; uma decisão nova é registrada em `docs/decisoes.md` no mesmo commit em que é implementada.
- Quando uma decisão couber ao autor (arquitetura, trade-offs, escopo), apresentar as opções com uma recomendação e aguardar a validação.

---

## 11. Controle de versão

- Repositório remoto: **https://github.com/srNeMeSes/GreyDetect** — todos os commits são enviados para ele.
- O repositório é **público**: nunca versionar dados do CrowdHuman, checkpoints, pesos treinados, credenciais ou binários de bibliotecas.
- Commits pequenos, um por etapa ou subetapa validada, com mensagem em pt-BR no imperativo (ex.: `Adiciona leitor de anotações .odgt`).
- Não enviar commits com build quebrado ou testes falhando sem que isso esteja declarado na mensagem.
- Mensagens de commits gerados com IA terminam com a linha de coautoria definida para a sessão.

---

## 12. Critérios de conclusão da primeira versão

A v1 é considerada funcional quando:

1. A arquitetura autoral estiver implementada e documentada (§4.2).
2. O pipeline de dados estiver validado (Fase 2).
3. O treinamento estiver operacional e reproduzível (Fases 5 e 6).
4. O modelo apresentar resultados mensuráveis no conjunto de teste (Fase 7).
5. A inferência gerar bounding boxes de pessoas em imagens e vídeos (Fase 8).
6. O desempenho de inferência estiver medido em hardware identificado.
7. O código compilar e os testes relevantes passarem.
8. As limitações conhecidas estiverem documentadas.

O rótulo "tempo real" exige um objetivo de desempenho definido e benchmarks reproduzíveis que o sustentem.

---

## 13. Estado atual

| Fase | Situação |
|---|---|
| 0 — Análise e planejamento | **Em andamento.** Ambiente inspecionado (§3.1). Pendentes: toolchain, versões, estrutura |
| 1–9 | Não iniciadas |
