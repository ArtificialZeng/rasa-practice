# rasa-practice

目录

    处理超出范围的消息
        1.创建超出范围的意图(nlu.yml)
        2.定义响应消息(domain.yml)
        3.创建超出范围规则(rules.yml)
        处理特定的超出范围消息
    Fallbacks
        NLU Fallback
            1.更新配置文件(config.yml)
            2.定义响应消息(domain.yml)
            3.创建NLU fallback规则(rules.yml)
        处理置信度较低的Action
            1.更新配置文件(config.yml)
            2.定义默认响应消息(domain.yml)
            3.自定义默认操作(actions.py)
        Two-Stage Fallback
            1.更新配置(config.yml)
            2.定义回退响应(domain.yml)
            3.定义两阶段回退(Two-Stage Fallback)规则
        切换到人工服务
    总结
    参考

处理超出范围的消息

为了避免用户失望，您可以处理您知道用户可能会问的问题，但您尚未实现用户目标。
1.创建超出范围的意图(nlu.yml)

您需要在NLU训练数据中定义out_of_scope的意图，并添加任何已知的超出范围请求作为训练示例，例如：

nlu:
- intent: out_of_scope
  examples: |
    - I want to order food
    - What is 2 + 2?
    - Who's the US President?


正如所有的意图一样，你应该从真实的对话中获取你的大多数例子。
2.定义响应消息(domain.yml)

您需要在域文件中定义一个超出范围的响应。使用utter_out_of_scope作为默认响应，如下所示：

responses:
  utter_out_of_scope:
  - text: Sorry, I can't handle that request.

  

3.创建超出范围规则(rules.yml)

最后，您需要为范围内和范围外的请求编写规则：

rules:
- rule: out-of-scope
  steps:
  - intent: out_of_scope
  - action: utter_out_of_scope

 

处理特定的超出范围消息

如果您注意到用户总是询问某些内容，你想要在将来将其转化为用户目标，您可以将这些作为单独的意图处理，让用户知道您已经理解了他们的消息，但还没有一个解决方案。例如，如果用户询问“我想在Rasa申请工作”，我们可以回答“我知道你在找工作，但恐怕我还不能应付这个技能。”
与out_of_scope意图示例类似，您需要创建新的意图，并列出训练示例、定义响应消息并创建规则。
Fallbacks

尽管Rasa会对未看到的消息进行泛化，但有些消息可能仍会收到较低的分类置信度。使用Fallbacks将有助于确保这些较低分类置信度的消息得到妥善处理，从而使助手可以选择使用默认消息响应，或尝试消除用户输入的歧义。
NLU Fallback

要处理NLU可信度较低的传入消息，请使用FallbackClassifier。使用此配置，当所有其他意图预测低于配置的置信阈值时，将预测意图nlu_fallback。然后，您可以编写一个规则，说明在预测nlu_fallback时bot应该做什么。
1.更新配置文件(config.yml)

要使用FallbackClassifier，请将其添加到NLU管道：

pipeline:
# other components
- name: FallbackClassifier
  threshold: 0.7


2.定义响应消息(domain.yml)

当消息被分类为较低的置信度时，通过添加响应来定义bo应发送的消息，如下：

responses:
  utter_please_rephrase:
  - text: I'm sorry, I didn't quite understand that. Could you rephrase?

  

3.创建NLU fallback规则(rules.yml)

当用户发送的消息置信度较低时，以下规则将要求用户重新措辞：

rules:
- rule: Ask the user to rephrase whenever they send a message with low NLU confidence
  steps:
  - intent: nlu_fallback
  - action: utter_please_rephrase



处理置信度较低的Action

由于用户可能会发送意料之外的消息，他们的行为可能会引导他们进入未知的对话路径。Rasa的机器学习策略（如TED Policy）被优化以处理这些未知路径。
若要处理机器学习策略无法以高置信度预测下一个Action的情况，即，如果在没有Policy具有置信度高于可配置阈值的下一个操作预测时，可以将Rule Policy配置为预测默认操作。
当Action可信度低时，您可以使用以下步骤配置应该运行的Action以及相应的可信度阈值：
1.更新配置文件(config.yml)

您需要将RulePolicy添加到config.yml中的policies中。默认情况下，规则策略附带以下设置：

policies:
- name: RulePolicy
  # Confidence threshold for the `core_fallback_action_name` to apply.
  # The action will apply if no other action was predicted with
  # a confidence >= core_fallback_threshold
  core_fallback_threshold: 0.4
  core_fallback_action_name: "action_default_fallback"
  enable_fallback_prediction: True


2.定义默认响应消息(domain.yml)

为了明确当动作可信度低于阈值时，bot将说什么，请定义一个响应utter_default：

responses:
  utter_default:
  - text: Sorry I didn't get that. Can you rephrase?

   

当action可信度低于阈值时，Rasa将运行action_default_fallback。这将发送响应utter_default，并恢复到导致回退的用户消息之前的会话状态，因此不会影响对未来操作的预测。
3.自定义默认操作(actions.py)

action_default_fallback是Rasa中的一个默认操作，它向用户发送utter_default响应。您可以创建自己的自定义操作以用作fallback（有关自定义操作的详细信息，请参阅Custom Actions）。下面的代码段是一个自定义操作的实现，该操作与action_default_fallback的操作相同，但分配一个不同的模板(响应)utter_fallback_template：

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.executor import CollectingDispatcher

class ActionDefaultFallback(Action):
    """Executes the fallback action and goes back to the previous state
    of the dialogue"""

    def name(self) -> Text:
        return ACTION_DEFAULT_FALLBACK_NAME

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(template="my_custom_fallback_template")

        # Revert user message which led to fallback.
        return [UserUtteranceReverted()]

  
Two-Stage Fallback

为了让bot有机会弄清楚用户想要什么，您通常会希望它通过问一些明确的问题来尝试消除用户消息的歧义。Two-Stage Fallback使用以下顺序处理多个阶段中的低NLU置信度：

    用户消息分类后置信度很低
    - 要求用户确认意图
    用户确认或否认意图
    - 如果他们确认了，谈话就会继续进行，就好像从一开始对意图进行了分类的置信度很高一样。不会采取进一步的回退步骤。
    - 如果他们否认，用户将被要求重新表述他们的消息。
    用户重新表述他们的意图
    - 如果消息被分类后有很高的置信度，则会话将继续，就好像用户从一开始就有此意图一样。
    - 如果用户消息重新措辞后的置信度仍然较低，则要求用户确认意图。
    用户确认或否认重新表述的意图
    - 如果他们确认，对话将继续，就好像用户从一开始就有这个意图一样。
    - 如果他们否认，最终的fallback action就会触发（例如，a handoff to a human）。

可以使用以下步骤启用Two-Stage-Fallback：
1.更新配置(config.yml)

将FallbackClassifier添加到pipeline中，并将RulePolicy添加到policies中：

pipeline:
# other components
- name: FallbackClassifier
  threshold: 0.7

policies:
# other policies
- RulePolicy

   

2.定义回退响应(domain.yml)

要定义bot如何要求用户重新表述其消息，请定义响应utter_ask_rephrase：

responses:
  utter_ask_rephrase:
  - text: I'm sorry, I didn't quite understand that. Could you rephrase?


Rasa提供了用于询问用户意图的默认实现，并要求用户重新措辞。要自定义这些操作的行为，请参阅有关默认操作的文档。
3.定义两阶段回退(Two-Stage Fallback)规则

将以下规则添加到训练数据中。此规则将确保每当接收到分类可信度较低的消息时，将激活Two-Stage-Fallback，在rules.yml定义以下规则：

rules:
- rule: Implementation of the Two-Stage-Fallback
  steps:
  - intent: nlu_fallback
  - action: action_two_stage_fallback
  - active_loop: action_two_stage_fallback

   

切换到人工服务

作为回退操作的一部分，您可能希望bot移交给人工代理，例如作为两阶段回退中的最终操作，或者当用户明确要求人工代理时。实现人工切换的一种简单方法是根据特定的bot或用户消息配置消息或语音通道，以切换它侦听的主机。
例如，作为两阶段回退的最后一个操作，bot可以询问用户，“您想被转移到人工助理吗？”如果他们说是，bot会向通道发送一条带有特定有效负载的消息，例如handoff_to_human。当通道看到此消息时，它将停止侦听Rasa服务器，并将一条消息发送到human通道，其中包含聊天对话的记录。
要实现从前端传递给人工，将取决于您使用的通道。您可以在Financial Demo和Helpdesk-Assistant StarterPack中看到一个使用聊天室频道自适应的示例实现。
总结

为了让您的助手较好地处理失败，您应该处理已知的“超出范围”的消息，并添加一个回退行为的表单。如果您想添加人工切换，您可以将其添加到备用设置中，或者作为备用设置的最后一步。以下是您需要对每个方法进行的更改的总结：

对于超出范围(out-of-scope)的意图：

    在NLU数据中为每个超出范围的意图添加训练示例
    定义超出范围的响应或操作
    为每个超出范围的意图定义规则
    将RulePolicy添加到config.yml

对于单步NLU回退：

    在config.yml中将FallbackClassifier添加到管道中
    定义回退响应或操作
    为nlu_fallback意图的定义规则
    将RulePolicy添加到config.yml

For handling low core confidence:

    在config.yml中为core fallback配置RulePolicy
    Optionally customize the fallback action you configure
    定义utter_default响应

For Two-Stage Fallback:

    在config.yml中将FallbackClassifier添加到pipeline中
    为nlu_fallback意图定义触发action_two_stage_fallback操作的规则
    在您的domain中定义out-of-scope的意图
    将RulePolicy添加到config.yml

切换到人工：

    Configure your front end to switch hosts
    Write a custom action (which could be your fallback action) to send the handoff payload
    Add a rule for triggering handoff (if not part of fallback)
    将RulePolicy添加到config.yml

参考

    官方文档
