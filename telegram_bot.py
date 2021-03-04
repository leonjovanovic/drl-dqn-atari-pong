
import logging

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
'''
# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)
'''

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
update_t = 1
context_t = 1
welcome_st = "P"

def info(iteration, max_iteration, mean_reward, loss):
    """Send a message when the command /start is issued."""
    st = 'Iteration %d/%d  Mean reward: %.2f Loss: %.4f' % (iteration, max_iteration, mean_reward, loss)
    #st = "Iteration " + str(iteration) + "/" + str(max_iteration) + " Mean reward: " + str(mean_reward) + " Loss: " + str(loss)
    update_t.message.reply_text(st)
    
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')
    global update_t
    global context_t
    update_t = update
    context_t = context
    
def ready():
    if update_t == 1:
        return False
    else:
        return True

def stop(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Stoping...')
    global update_t
    global context_t
    update_t = 1
    context_t = 1

def main():
    print("Bot started")
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("1415708047:AAGwFzLqd3C4Lq2vSqNO0151_pr8IpRwwNE")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("stop", stop))    

    # on noncommand i.e message - echo the message on Telegram
    #dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()
    
    #info(update= Update, context= CallbackContext)

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    #updater.idle()

if __name__ == "__main__":
    main()