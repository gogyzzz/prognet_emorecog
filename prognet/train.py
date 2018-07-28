#xxx train.py

from toolz import curry

# on pytorch

@curry
def validate_loop_lazy(name, __validate, loader):

    losses = [0.0] * len(loader)
    scores = [0.0] * len(loader)

    for i, batch in enumerate(loader):

        losses[i], scores[i]= __validate(batch)

    if len(loader) > 1:
        score = sum(scores[:-1])/(len(scores[:-1]))
        loss = sum(losses[:-1])/(len(losses[:-1]))

    else:
        score = scores[0]
        loss = losses[0]

    print('[%s] score: %.3f, loss: %.3f'
            %(name, score, loss))
        
    return loss, score


def train(model, loader, _valid_lazy, valid_loop, crit, optim):

    best_valid_score = 0.0
    best_model = model

    for epoch in range(ephs):
        for i, batch in enumerate(loader):

            inputs = batch[0]
            targets = batch[1]
            #print('batch',batch)

            optim.zero_grad()
            model.train() # autograd on

            train_loss = crit(model(inputs), targets)
            train_loss.backward()

            optim.step()
            model.eval() # autograd off

            __val_lz = _valid_lazy(model=model)

        print('[train] %5dth epoch, loss: %.3f'
                %(epoch, train_loss.data[0]))

        valid_loss, valid_score = valid_loop(__validate=__val_lz)

        if valid_score > best_valid_score:

            best_valid_score = valid_score

            print('[valid] bestscore: %.3f, loss: %.3f'
            %(valid_score, valid_loss))

            best_model = model

    print('Finished Training')

    return best_model

