function [mae,rmse,mape,error]=calc_error(x1,x2)

 error=x2-x1;  %�������
 rmse=sqrt(mean(error.^2));
 disp(['1.��������(RMSE)��',num2str(rmse)])

 mae=mean(abs(error));
disp(['2.ƽ��������MAE����',num2str(mae)])

 mape=mean(abs(error)/x1);
 disp(['3.ƽ����԰ٷ���MAPE����',num2str(mape*100),'%'])

end

