def label_to_int(string_label):
    if string_label == 'estrella_tp': return 1
    if string_label == 'rectangulo_tp': return 2
    if string_label == 'triangulo_tp':
        return 3

    else:
        raise Exception('unkown class_label')



def int_to_label(string_label):
    if string_label == 1: return 'estrella_tp'
    if string_label == 2: return 'rectangulo_tp'
    if string_label == 3:
        return 'triangulo_tp'
    else:
        raise Exception('unkown class_label')

