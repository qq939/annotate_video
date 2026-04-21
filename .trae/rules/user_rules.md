开发python脚本必须遵循规则：
ANSWER IN CHINESE!
.trae/reference/ref.txt（如果没有请新建）里面是需要参考的github地址或接口文档，如果你认为有哪些可以参考，在网上搜索补充到该文件中(mode: add);
global参数前置到py文件最上边，并且有具体使用位置精确到行的注释;
灵活使用agent skill和mcp来完成任务;
完成所有任务清单，完成之前不要退出;
TDD模式，每个任务开始前先写测试脚本，测试必须有超时机制，脚本必须通过测试才算完成任务;
将user_rules.md文件中的所有规则都保存在：.trae/rules/user_rules.md中;
如果有git仓库，先暂存本地修改，然后git pull，然后再继续下面的步骤;
Create python venv evironment(use command: uv), and install python packages in requirements.txt, with -i https://pypi.tuna.tsinghua.edu.cn/simple/
每次对话后都要确保python的import不缺失，requirements.txt里的模块不缺失，requirements.txt里面不要写版本号，requirements_{python version}.txt里面是带版本号的模块;
每次对话后都要git push to origin:main，commit内容就是我说的那句话。user.email="939342547@qq.com", user.name="qq939", remote=https://github.com/qq939/annotate_video, branch=main;
git add .trae/rules/project_rules.md
git add .trae/rules/user_rules.md
如果git推送到远端失败，rebase并且push --force-with-lease