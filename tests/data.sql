INSERT INTO user (username,
                  password,
                  document_number,
                  sens_document_number,
                  non_sens_document_number,
                  visualisation_method,
                  clf_method)
VALUES ('test', 'pbkdf2:sha256:50000$TCI4GzcX$0de171a4f4dac32e3364c7ddc7c14f3e2fa61f2d17574483f7ffbb431b4acb2f', 1, 1, 1, 'ELI5', 'LR'),
       ('other', 'pbkdf2:sha256:50000$kJPKsz6N$d2d4784f1b030a9761f5ccaeeaca413f27f2ecb76d6168407af962ddce849f79', 100, 100, 100, 'LIME', 'LR');

