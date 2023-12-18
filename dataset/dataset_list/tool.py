def dataset_generation(dataset_one):
    if dataset_one == 'climate':
        from dataset.dataset_list.climate import climate
        data = climate()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'emo':
        from dataset.dataset_list.emo import emo
        data = emo()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'emotion':
        from dataset.dataset_list.emotion import emotion
        data = emotion()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'ethos':
        from dataset.dataset_list.ethos import ethos
        data = ethos()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'financial_phrasebank':
        from dataset.dataset_list.financial_phrasebank import financial_phrasebank
        data = financial_phrasebank()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'hate_speech18':
        from dataset.dataset_list.hate_speech18 import hate_speech18
        data = hate_speech18()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'glue_cola':
        from dataset.dataset_list.glue_cola import glue_cola
        data = glue_cola()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'glue_mnli':
        from dataset.dataset_list.glue_mnli import glue_mnli
        data = glue_mnli()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'glue_mrpc':
        from dataset.dataset_list.glue_mrpc import glue_mrpc
        data = glue_mrpc()
        dataset = data.dataset_to_DataFrame()



    elif dataset_one == 'glue_qnli':
        from dataset.dataset_list.glue_qnli import glue_qnli
        data = glue_qnli()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'glue_qqp':
        from dataset.dataset_list.glue_qqp import glue_qqp
        data = glue_qqp()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'glue_rte':
        from dataset.dataset_list.glue_rte import glue_rte
        data = glue_rte()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'glue_sst2':
        from dataset.dataset_list.glue_sst2 import glue_sst2
        data = glue_sst2()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'glue_wnli':
        from dataset.dataset_list.glue_wnli import glue_wnli
        data = glue_wnli()
        dataset = data.dataset_to_DataFrame()

    elif dataset_one == 'health_fact':
        from dataset.dataset_list.health_fact import health_fact
        data = health_fact()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'imdb':
        from dataset.dataset_list.imdb import imdb
        data = imdb()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'medical_questions_pairs':
        from dataset.dataset_list.medical_questions_pairs import medical_questions_pairs
        data = medical_questions_pairs()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'paws':
        from dataset.dataset_list.paws import paws
        data = paws()
        dataset = data.dataset_to_DataFrame()

    elif dataset_one == 'poem_sentiment':
        from dataset.dataset_list.poem_sentiment import poem_sentiment
        data = poem_sentiment()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'scitail':
        from dataset.dataset_list.scitail import scitail
        data = scitail()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'sick':
        from dataset.dataset_list.sick import sick
        data = sick()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'sms_spam':
        from dataset.dataset_list.sms_spam import sms_spam
        data = sms_spam()
        dataset = data.dataset_to_DataFrame()


    elif dataset_one == 'trec':
        from dataset.dataset_list.trec import trec
        data = trec()
        dataset = data.dataset_to_DataFrame()


    else:
        from dataset.dataset_list.climate import climate
        data = climate()
        dataset = data.dataset_to_DataFrame()

    return dataset
