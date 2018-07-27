##### train.py
def validate(model, name, loader, crit):

    loss = 0.0
    score = 0.0

    for i, batch in enumerate(loader):
        inputs, targets, sample_wgts = batch
        outputs = model(inputs)
        loss += crit(outputs, targets)
        # 
        # score+=score_func(outputs,targets,sample_wgts)
        ##score.py##

    loss = loss/(i+1)
    score = score/(i+1)

    print('[%s] score: %.3f, loss: %.3f'
            %(name, score, loss))
        
    return loss/(i+1), score/(i+1)


def train(model, loader, valid_func, crit, optim):

    best_valid_score = 0.0
    best_model = model

    for epoch in range(ephs):
        for i, batch in enumerate(loader):
            
            inputs, targets = batch

            optim.zero_grad()
            model.train() # autograd on

            train_loss = crit(model(inputs), targets)
            train_loss.backward()

            print('[train] %5dth epoch, %5dth batch. loss: %.3f'
                    %(epoch, i, train_loss.data[0]))

            optim.step()
            model.eval() # autograd off
            valid_loss, valid_score = valid_func(model)

            if valid_score > best_valid_score:

                best_valid_uar = valid_uar

                print('[valid] bestscore: %.3f, loss: %.3f'
                %(valid_score, valid_loss))

                best_model = model

            print('Finished Training')

            return best_model
