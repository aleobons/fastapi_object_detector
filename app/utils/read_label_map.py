def read_label_map(label_map_path: str) -> dict:
    '''Lê o arquivo com o mapeamento de rótulos

    O mapeamento deve estar no protobuf format

    Args:
        label_map_path: uma string com o caminho para o arquivo com o mapeamento de rótulos

    Returns:
        items: um dicionário contendo o id do objeto como chave e o nome do objeto como valor

    Raises:
        Sem raises
    '''

    # inicializa as variáveis
    item_id = None
    item_name = None
    items = {}

    with open(label_map_path, "r") as file:
        for line in file:
            # procura linhas que contenham o ID e o display_name do objeto
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1])
            elif "display_name" in line:
                item_name = line.split(":", 1)[1].replace("\"", "").strip()

            # verifica se já é possível incluir um novo objeto no dicionário de items
            if item_id is not None and item_name is not None:
                items[item_id] = item_name

                # reinicializa as variáveis
                item_id = None
                item_name = None

    return items
