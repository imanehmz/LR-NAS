from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
      'none',
      'max_pool_3x3',
      'avg_pool_3x3',

      'skip_connect_u',
      'skip_connect_f',
      'skip_connect_b',

      'sep_conv_3x3_u',
      'sep_conv_3x3_br',
    'sep_conv_3x3_fr',
    'sep_conv_3x3_fa',

    'sep_conv_5x5_u',
    'sep_conv_5x5_br',
    'sep_conv_5x5_fr',
    'sep_conv_5x5_fa',

    'dil_conv_3x3_u',
    'dil_conv_3x3_br',
    'dil_conv_3x3_fr',
    'dil_conv_3x3_fa',

    'dil_conv_5x5_u',
    'dil_conv_5x5_br',
    'dil_conv_5x5_fr',
    'dil_conv_5x5_fa',

]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

LR_NAS1 = Genotype(
normal=[
('skip_connect', 0),
 ('skip_connect', 1),
 ('sep_conv_3x3_u', 0),
 ('skip_connect', 1),
 ('sep_conv_3x3_u', 0),
 ('sep_conv_3x3_fr', 1),
 ('skip_connect', 2),
 ('skip_connect', 0)
], 
normal_concat=range(2, 6),
 reduce=[
('skip_connect', 1),
 ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_5x5_br', 3), ('skip_connect', 3), ('skip_connect', 0)], reduce_concat=range(2, 6))



DARTS_maxpool = Genotype(normal=[('max_pool_3x3', 0), ('skip_connect', 1), 
('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
 ('max_pool_3x3', 0), ('sep_conv_3x3_u', 1)], normal_concat=range(2, 6),
 reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
 ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2),
 ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

BIODARTS2 = Genotype(normal=[('skip_connect_f', 0), ('skip_connect_f', 1), ('sep_conv_3x3_fr', 1), ('sep_conv_3x3_u', 2), ('sep_conv_3x3_u', 0), ('skip_connect_f', 1), ('sep_conv_3x3_fr', 1), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect_u', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5_u', 3)], reduce_concat=range(2, 6))

BIODARTS = Genotype(normal=[('skip_connect_f', 0),
                            ('skip_connect_f', 1),
                            ('sep_conv_3x3_fr', 1),
                            ('sep_conv_3x3_u', 2), 
                            ('sep_conv_3x3_u', 0), 
                            ('skip_connect_f', 1), 
                            ('sep_conv_3x3_fr', 1), 
                            ('max_pool_3x3', 0)], 
                    normal_concat=range(2, 6), 
                    reduce=[('max_pool_3x3', 0), 
                            ('skip_connect_u', 1), 
                            ('max_pool_3x3', 0), 
                            ('max_pool_3x3', 1), 
                            ('max_pool_3x3', 0), 
                            ('max_pool_3x3', 1), 
                            ('max_pool_3x3', 0), 
                            ('dil_conv_5x5_u', 3)], 
                    reduce_concat=range(2, 6))

DARTS = LR_NAS1


