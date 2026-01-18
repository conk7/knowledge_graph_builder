С момента публикации «Attention Is All You Need» в 2017 году архитектура Transformer стала универсальным строительным элементом практически во всех областях ИИ: язык, зрение, аудио, биология, физика, робототехника.

### Каноническая структура (2017)
Каждый слой состоит из двух основных суб-слоёв:
1. Multi-Head Self-Attention + residual connection + LayerNorm
2. Position-wise Feed-Forward Network + residual + LayerNorm

Encoder и decoder отличаются лишь типами внимания:
- Encoder: только self-attention
- Decoder: masked self-attention + encoder-decoder attention

### Три основные ветви эволюции
1. **Encoder-only** (BERT, RoBERTa, DeBERTa, ELECTRA)  
   → понимание текста, классификация, NER, поиск

2. **Decoder-only** (GPT-1/2/3/4, LLaMA, PaLM, Grok, Mistral, Qwen)  
   → генерация текста, in-context learning, агенты

3. **Encoder-Decoder** (T5, BART, UL2, Flan-T5, NLLB)  
   → перевод, суммаризация, вопрос-ответ, text-to-text задачи

### Модификации и улучшения 2018–2025
| Год | Улучшение                        | Эффект                              |
|-----|----------------------------------|-------------------------------------|
| 2018| Pre-LN вместо Post-LN            | Стабильнее обучение глубоких моделей|
| 2019| ALiBi, Rotary (RoPE)             | Экстраполяция на длинные контексты  |
| 2020| Sandwich-LN, RMSNorm             | Быстрее, стабильнее                 |
| 2021| FlashAttention, xFormers         | 3–7× ускорение инференса            |
| 2023| Grouped-Query Attention (LLaMA-2)| Лучший компромисс скорость/качество |
| 2024| Mamba, RWKV, RetNet              | Линейная сложность по длине         |

### Почему Transformer так долго остаётся непобеждённым
- Параллелизм обучения
- Гибкость (можно вставлять любые адаптеры, LoRA, prefix-tuning)
- Богатый выразительный язык (сотни готовых модификаций)
- Масштабируемость до триллионов параметров

Даже самые перспективные альтернативы 2024–2025 (Mamba-2, Liquid CNN, Kolmogorov-Arnold Networks) пока либо уступают в качестве на языковых задачах, либо являются гибридами с Transformer-блоками.

См. также: [[Attention]], [[GPT-3]], [[BERT]]