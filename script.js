/**
 * MAPEAMENTO COMPLETO (Baseado no seu CSV e nos atributos originais)
 */
const MAPEAMENTO_IA = {
    "cap-shape": ['b', 'c', 'x', 'f', 'k', 's'],
    "cap-surface": ['f', 'g', 'y', 's'],
    "cap-color": ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
    "bruises": ['t', 'f'],
    "odor": ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
    "gill-attachment": ['a', 'd', 'f', 'n'],
    "gill-spacing": ['c', 'w', 'd'],
    "gill-size": ['b', 'n'],
    "gill-color": ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
    "stalk-shape": ['e', 't'],
    "stalk-root": ['b', 'c', 'u', 'e', 'z', 'r', '?'],
    "stalk-surface-above-ring": ['f', 'y', 'k', 's'],
    "stalk-surface-below-ring": ['f', 'y', 'k', 's'],
    "stalk-color-above-ring": ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    "stalk-color-below-ring": ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    "veil-type": ['p', 'u'],
    "veil-color": ['n', 'o', 'w', 'y'],
    "ring-number": ['n', 'o', 't'],
    "ring-type": ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
    "spore-print-color": ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
    "population": ['a', 'c', 'n', 's', 'v', 'y'],
    "habitat": ['g', 'l', 'm', 'p', 'u', 'w', 'd']
};

let arquivoCsv = null;

document.getElementById('csv-file').onchange = (e) => {
    arquivoCsv = e.target.files[0];
    document.getElementById('btn-treinar').disabled = false;
    document.getElementById('status').innerText = "Arquivo selecionado: " + arquivoCsv.name;
};

async function iniciarTreino() {
    tfvis.visor().open();
    const status = document.getElementById('status');
    status.innerText = "Processando dados...";

    const texto = await arquivoCsv.text();
    const linhas = texto.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
    
    // ATENÇÃO: Usando ';' como separador conforme seu arquivo
    const cabecalho = linhas[0].split(';').map(c => c.trim().toLowerCase());

    const xData = [];
    const yData = [];

    for (let i = 1; i < linhas.length; i++) {
        const colunas = linhas[i].split(';').map(c => c.trim().toLowerCase());
        if (colunas.length < cabecalho.length) continue;

        // Label: primeira coluna (class)
        yData.push(colunas[0] === 'p' ? 1 : 0);

        // Features: colunas 1 em diante
        let vetorLinha = [];
        for (let j = 1; j < colunas.length; j++) {
            const nomeAttr = cabecalho[j];
            const valor = colunas[j];
            const opcoes = MAPEAMENTO_IA[nomeAttr];
            
            if (opcoes) {
                const oneHot = new Array(opcoes.length).fill(0);
                const idx = opcoes.indexOf(valor);
                if (idx !== -1) oneHot[idx] = 1;
                vetorLinha.push(...oneHot);
            }
        }
        xData.push(vetorLinha);
    }

    if (xData.length === 0) {
        status.innerText = "Erro: Dados não processados. Verifique se o separador é ';' ";
        return;
    }

    const xTensor = tf.tensor2d(xData, [xData.length, xData[0].length]);
    const yTensor = tf.tensor2d(yData, [yData.length, 1]);

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [xData[0].length] })); // Número de neurônios
    model.add(tf.layers.dense({ units: 1, activation: 'relu' }));
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    status.innerText = "Treinando...";
    
    await model.fit(xTensor, yTensor, {
        epochs: 15,
        batchSize: 128,
        validationSplit: 0.,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Métricas', tab: 'Treino' },
            ['loss', 'acc'],
            {
                callback: ['onEpochEnd']
            }
        )
    });

    status.innerText = "✅ Treino Concluído! Modelo salvo.";
    await model.save('localstorage://modelo-cogumelos');
}

document.getElementById('btn-treinar').onclick = iniciarTreino;